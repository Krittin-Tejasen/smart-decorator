-- Run this FIRST in the Supabase SQL editor (Dashboard -> SQL Editor) and read
-- the result. It changes nothing: it only lists what protects the tables and
-- the storage bucket the "saved rooms" feature uses today.
--
-- Why: the app's anon key can currently list the `room-images` bucket and read
-- `room_captures`, i.e. some policy is more open than the per-user model in
-- migrations/20260927000000_per_user_saved_rooms.sql. Old policies are OR-ed
-- with new ones, so an open one left behind would defeat the new ones.

-- 1) Row level security on/off per table
select schemaname, tablename, rowsecurity as rls_enabled
from pg_tables
where schemaname = 'public'
  and tablename in ('room_designs', 'design_furniture', 'room_captures', 'ikea_furniture')
order by tablename;

-- 2) Every policy on those tables and on storage objects
select schemaname, tablename, policyname, cmd, roles, qual, with_check
from pg_policies
where (schemaname = 'public' and tablename in ('room_designs', 'design_furniture', 'room_captures', 'ikea_furniture'))
   or (schemaname = 'storage' and tablename = 'objects')
order by schemaname, tablename, policyname;

-- 3) Are anonymous sign-ins allowed? (Dashboard -> Authentication -> Sign In / Providers
--    -> "Allow anonymous sign-ins".) It cannot be read from SQL; check the toggle.
