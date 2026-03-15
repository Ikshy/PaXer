-- docker/postgres/init.sql
-- Run once when the Postgres container is first created.
-- Creates the cthmp database and grants privileges.
-- (The SQLAlchemy models create the actual tables on app startup.)

\c cthmp

-- Enable useful extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Ensure the app user has full privileges
GRANT ALL PRIVILEGES ON DATABASE cthmp TO cthmp_user;
GRANT ALL ON SCHEMA public TO cthmp_user;
