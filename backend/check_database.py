#!/usr/bin/env python3
"""
Script to check and initialize database schema
"""

import asyncio
import asyncpg
import os
from pathlib import Path

async def check_database():
    """Check if database tables exist and create them if needed"""
    
    # Database connection
    database_url = os.getenv("DATABASE_URL", "postgresql://towerco:password@localhost:5432/towerco_aiops")
    
    try:
        # Connect to database
        conn = await asyncpg.connect(database_url)
        print("✅ Connected to database")
        
        # Check if sites table exists
        result = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'sites'
            );
        """)
        
        if result:
            print("✅ Sites table exists")
            
            # Check if there are any sites
            count = await conn.fetchval("SELECT COUNT(*) FROM sites")
            print(f"📊 Found {count} sites in database")
            
        else:
            print("❌ Sites table does not exist")
            print("🔧 Please run the database initialization script:")
            print("   docker exec -it compose-postgres-1 psql -U towerco -d towerco_aiops -f /docker-entrypoint-initdb.d/01-init-timescaledb.sql")
            
        # Check other important tables
        tables_to_check = [
            'tenants', 'network_metrics', 'energy_metrics', 
            'events', 'kpi_definitions', 'kpi_values'
        ]
        
        for table in tables_to_check:
            exists = await conn.fetchval(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = '{table}'
                );
            """)
            
            if exists:
                print(f"✅ Table '{table}' exists")
            else:
                print(f"❌ Table '{table}' missing")
        
        await conn.close()
        print("✅ Database check completed")
        
    except Exception as e:
        print(f"❌ Database error: {e}")

if __name__ == "__main__":
    asyncio.run(check_database())
