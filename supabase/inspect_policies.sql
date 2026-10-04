-- Run this FIRST in the Supabase SQL editor (Dashboard -> SQL Editor -> New
-- query -> paste -> Run) and send Claude the result rows. It is ONE query, so
-- the editor shows everything (it only displays the last statement's result).
-- It changes nothing: it only lists what protects the tables and the storage
-- bucket the "saved rooms" feature uses today.
--
-- Why: some policy is more open than the per-user model in
-- migrations/20260927000000_per_user_saved_rooms.sql (the app could list the
-- `room-images` bucket and read `room_captures` without logging in, and upload
-- to the bucket as an anonymous user). Old policies are OR-ed with new ones, so
-- an open one left behind would defeat the new ones.
--
-- Rows: kind = 'table' (is row level security on?) or 'policy' (who may do what).

select 'table' as kind,
       schemaname || '.' || tablename as object,
       case when rowsecurity then 'RLS ON' else 'RLS OFF' end as detail,
       null::text as cmd,
       null::text as roles,
       null::text as using_expr,
       null::text as check_expr
from pg_tables
where (schemaname = 'public' and tablename in ('room_designs', 'design_furniture', 'room_captures', 'ikea_furniture'))
   or (schemaname = 'storage' and tablename = 'objects')

union all

select 'policy',
       schemaname || '.' || tablename || ' :: ' || policyname,
       null,
       cmd,
       roles::text,
       qual,
       with_check
from pg_policies
where (schemaname = 'public' and tablename in ('room_designs', 'design_furniture', 'room_captures', 'ikea_furniture'))
   or (schemaname = 'storage' and tablename = 'objects')

order by kind desc, object;

-- Also worth a look (cannot be read from SQL): Dashboard -> Authentication ->
-- Sign In / Providers -> "Allow anonymous sign-ins" must be ON (it is now).
