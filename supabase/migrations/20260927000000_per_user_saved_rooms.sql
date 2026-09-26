-- Per-user saved rooms (max 5 each), for the "Save Room Data" flow.
--
-- Before running:
--   1. Dashboard -> Authentication -> Sign In / Providers -> turn ON
--      "Allow anonymous sign-ins" (the app signs users in anonymously, no login screen).
--   2. Run supabase/inspect_policies.sql and check no open (anon) policy is left
--      on room_designs / room_captures / storage.objects for the room-images bucket.
--
-- Run in: Dashboard -> SQL Editor. Safe to run more than once.
--
-- What this does
--   * room_designs gets an owner (user_id = the anonymous user's auth.uid()).
--   * A user can read / insert / delete only their own rows, and can hold at most
--     max_saved_designs() rows (5). The limit lives in the database, so the app
--     cannot be tricked into exceeding it. To change it later (paid tiers),
--     edit max_saved_designs().
--   * Only the owner can read / upload / delete files under <their uid>/ in the
--     room-images bucket (private).
--   * room_captures (LiDAR / AR scan data) gets the same owner-only rules.
--
-- Rows written before this migration have no owner (user_id is null) and become
-- invisible to every app user. See the optional clean-up at the bottom.

-- ── 1. Ownership ───────────────────────────────────────────────────────────

alter table public.room_designs
  add column if not exists user_id uuid
    references auth.users (id) on delete cascade
    default auth.uid();

-- A row now exists only because the user saved it.
alter table public.room_designs alter column is_saved set default true;

create index if not exists room_designs_user_created_idx
  on public.room_designs (user_id, created_at desc);

-- ── 2. The 5-room limit ────────────────────────────────────────────────────

create or replace function public.max_saved_designs()
returns integer
language sql
immutable
as $$ select 5 $$;

create or replace function public.enforce_saved_design_limit()
returns trigger
language plpgsql
as $$
begin
  if new.user_id is null then
    raise exception 'user_id is required' using errcode = '23502';
  end if;

  -- Two saves at the same instant must not both squeeze in under the limit.
  perform pg_advisory_xact_lock(hashtext(new.user_id::text));

  if (select count(*) from public.room_designs where user_id = new.user_id)
       >= public.max_saved_designs() then
    -- The app looks for this exact message.
    raise exception 'save_limit_reached'
      using errcode = 'P0001',
            hint = format('At most %s saved rooms per user', public.max_saved_designs());
  end if;

  return new;
end;
$$;

drop trigger if exists room_designs_limit on public.room_designs;
create trigger room_designs_limit
  before insert on public.room_designs
  for each row execute function public.enforce_saved_design_limit();

-- ── 3. Row level security: room_designs ────────────────────────────────────

alter table public.room_designs enable row level security;

-- Start from a known state: drop whatever policies exist on this table (the old
-- backend wrote with the service key and needed none), then add ours.
do $$
declare r record;
begin
  for r in select policyname from pg_policies
           where schemaname = 'public' and tablename = 'room_designs' loop
    execute format('drop policy %I on public.room_designs', r.policyname);
  end loop;
end $$;

create policy sd_room_designs_select_own on public.room_designs
  for select to authenticated using (user_id = auth.uid());

create policy sd_room_designs_insert_own on public.room_designs
  for insert to authenticated with check (user_id = auth.uid());

create policy sd_room_designs_delete_own on public.room_designs
  for delete to authenticated using (user_id = auth.uid());

-- ── 4. Storage: room-images bucket, one folder per user ────────────────────
-- Files are stored as <auth.uid()>/<timestamp>/generated.png and source.jpg.

drop policy if exists sd_room_images_select_own on storage.objects;
drop policy if exists sd_room_images_insert_own on storage.objects;
drop policy if exists sd_room_images_delete_own on storage.objects;

create policy sd_room_images_select_own on storage.objects
  for select to authenticated
  using (bucket_id = 'room-images' and (storage.foldername(name))[1] = auth.uid()::text);

create policy sd_room_images_insert_own on storage.objects
  for insert to authenticated
  with check (bucket_id = 'room-images' and (storage.foldername(name))[1] = auth.uid()::text);

create policy sd_room_images_delete_own on storage.objects
  for delete to authenticated
  using (bucket_id = 'room-images' and (storage.foldername(name))[1] = auth.uid()::text);

-- ── 5. room_captures (LiDAR / AR scan data): owner only ────────────────────

alter table public.room_captures
  alter column user_id set default auth.uid();

alter table public.room_captures enable row level security;

do $$
declare r record;
begin
  for r in select policyname from pg_policies
           where schemaname = 'public' and tablename = 'room_captures' loop
    execute format('drop policy %I on public.room_captures', r.policyname);
  end loop;
end $$;

create policy sd_room_captures_select_own on public.room_captures
  for select to authenticated using (user_id = auth.uid());

create policy sd_room_captures_insert_own on public.room_captures
  for insert to authenticated with check (user_id = auth.uid());

create policy sd_room_captures_delete_own on public.room_captures
  for delete to authenticated using (user_id = auth.uid());

-- ── Optional clean-up of pre-migration data (NOT run automatically) ────────
-- Every generation used to be stored, saved or not (4 designs / 30 furniture
-- rows / ~3 MB of images at the time of writing). They have no owner now and
-- nobody can see them. To remove the rows:
--
--   delete from public.design_furniture;
--   delete from public.room_designs where user_id is null;
--
-- The image files under designs/ in the room-images bucket are NOT removed by
-- SQL (deleting storage.objects rows leaves the files behind); delete that
-- folder from Dashboard -> Storage -> room-images, or ask Claude to do it
-- through the API.
