-- Raise the per-user saved-room limit from 5 to 50.
--
-- Run in: Dashboard -> SQL Editor. Safe to run more than once.
--
-- This only redefines max_saved_designs() (used by the enforce_saved_design_limit()
-- trigger from 20260927000000_per_user_saved_rooms.sql) — no other schema change.
-- Keep lib/core/services/design_repository.dart's `savedDesignLimit` constant in
-- sync; it's a client-side pre-check only, the trigger below is the real limit.

create or replace function public.max_saved_designs()
returns integer
language sql
immutable
as $$ select 100 $$;
