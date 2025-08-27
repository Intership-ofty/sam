-- =============================================================================
-- TOWERCO AIOPS - DATABASE INITIALIZATION SCRIPT
-- Initialize TimescaleDB with extensions and schema for AIOps platform
-- =============================================================================

-- Create TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS uuid-ossp;

-- Create Keycloak database
CREATE DATABASE keycloak;

-- Connect to main database
\c towerco_aiops;

-- =============================================================================
-- CORE TABLES
-- =============================================================================

-- Sites table - Central registry of all telecom sites
CREATE TABLE sites (
    site_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    site_code VARCHAR(50) UNIQUE NOT NULL,
    site_name VARCHAR(200) NOT NULL,
    site_type VARCHAR(50) NOT NULL, -- BTS, NODEBS, ENODEB, GNODEB, etc.
    latitude DECIMAL(10,8),
    longitude DECIMAL(11,8),
    address TEXT,
    region VARCHAR(100),
    country VARCHAR(100),
    tenant_id UUID NOT NULL,
    technology JSONB, -- 2G, 3G, 4G, 5G capabilities
    energy_config JSONB, -- Power, battery, generator config
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Tenants table - Multi-tenant support for different MNOs
CREATE TABLE tenants (
    tenant_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_name VARCHAR(100) NOT NULL UNIQUE,
    tenant_code VARCHAR(20) NOT NULL UNIQUE,
    contact_email VARCHAR(255),
    sla_targets JSONB, -- SLA thresholds and targets
    billing_config JSONB,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =============================================================================
-- TIME-SERIES TABLES
-- =============================================================================

-- Network performance metrics (hypertable)
CREATE TABLE network_metrics (
    time TIMESTAMP WITH TIME ZONE NOT NULL,
    site_id UUID NOT NULL,
    tenant_id UUID NOT NULL,
    technology VARCHAR(10) NOT NULL, -- 2G, 3G, 4G, 5G
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15,6),
    unit VARCHAR(20),
    quality_score DECIMAL(3,2), -- 0.00 to 1.00
    metadata JSONB,
    PRIMARY KEY (time, site_id, metric_name)
);

-- Convert to hypertable with 1-day chunk interval
SELECT create_hypertable('network_metrics', 'time', chunk_time_interval => INTERVAL '1 day');

-- Energy metrics (hypertable)
CREATE TABLE energy_metrics (
    time TIMESTAMP WITH TIME ZONE NOT NULL,
    site_id UUID NOT NULL,
    tenant_id UUID NOT NULL,
    energy_type VARCHAR(50) NOT NULL, -- GRID, BATTERY, GENERATOR, SOLAR
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15,6),
    unit VARCHAR(20),
    efficiency_score DECIMAL(3,2),
    metadata JSONB,
    PRIMARY KEY (time, site_id, energy_type, metric_name)
);

SELECT create_hypertable('energy_metrics', 'time', chunk_time_interval => INTERVAL '1 day');

-- Events stream (hypertable)
CREATE TABLE events (
    time TIMESTAMP WITH TIME ZONE NOT NULL,
    event_id UUID DEFAULT uuid_generate_v4(),
    site_id UUID,
    tenant_id UUID,
    event_type VARCHAR(100) NOT NULL, -- ALARM, FAULT, MAINTENANCE, etc.
    severity VARCHAR(20) NOT NULL, -- CRITICAL, MAJOR, MINOR, WARNING, INFO
    source_system VARCHAR(100) NOT NULL,
    title VARCHAR(500) NOT NULL,
    description TEXT,
    impact_assessment TEXT,
    correlation_id UUID,
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by VARCHAR(255),
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    resolved BOOLEAN DEFAULT FALSE,
    resolved_by VARCHAR(255),
    resolved_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB,
    PRIMARY KEY (time, event_id)
);

SELECT create_hypertable('events', 'time', chunk_time_interval => INTERVAL '1 day');

-- =============================================================================
-- KPI TABLES
-- =============================================================================

-- KPI definitions and configurations
CREATE TABLE kpi_definitions (
    kpi_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    kpi_name VARCHAR(100) NOT NULL UNIQUE,
    kpi_category VARCHAR(50) NOT NULL, -- NETWORK, ENERGY, FINANCIAL, OPERATIONAL
    calculation_formula TEXT NOT NULL,
    unit VARCHAR(20),
    target_value DECIMAL(15,6),
    warning_threshold DECIMAL(15,6),
    critical_threshold DECIMAL(15,6),
    calculation_interval INTERVAL DEFAULT INTERVAL '5 minutes',
    enabled BOOLEAN DEFAULT TRUE,
    tenant_specific BOOLEAN DEFAULT FALSE,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Calculated KPI values (hypertable)
CREATE TABLE kpi_values (
    time TIMESTAMP WITH TIME ZONE NOT NULL,
    kpi_id UUID NOT NULL,
    site_id UUID,
    tenant_id UUID,
    kpi_value DECIMAL(15,6) NOT NULL,
    target_value DECIMAL(15,6),
    deviation_pct DECIMAL(5,2),
    quality_score DECIMAL(3,2),
    trend VARCHAR(20), -- IMPROVING, STABLE, DEGRADING
    metadata JSONB,
    PRIMARY KEY (time, kpi_id, COALESCE(site_id, '00000000-0000-0000-0000-000000000000'::UUID))
);

SELECT create_hypertable('kpi_values', 'time', chunk_time_interval => INTERVAL '1 hour');

-- =============================================================================
-- AIOPS TABLES
-- =============================================================================

-- Correlation rules and patterns
CREATE TABLE correlation_rules (
    rule_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    rule_name VARCHAR(200) NOT NULL,
    rule_type VARCHAR(50) NOT NULL, -- TIME_BASED, PATTERN_BASED, ML_BASED
    conditions JSONB NOT NULL,
    actions JSONB NOT NULL,
    confidence_threshold DECIMAL(3,2) DEFAULT 0.8,
    enabled BOOLEAN DEFAULT TRUE,
    tenant_id UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Root cause analysis results (hypertable)
CREATE TABLE rca_analysis (
    time TIMESTAMP WITH TIME ZONE NOT NULL,
    analysis_id UUID DEFAULT uuid_generate_v4(),
    incident_id UUID NOT NULL,
    root_cause TEXT,
    confidence_score DECIMAL(3,2),
    affected_sites UUID[],
    impact_assessment JSONB,
    recommended_actions JSONB,
    analysis_duration INTERVAL,
    algorithm_used VARCHAR(100),
    metadata JSONB,
    PRIMARY KEY (time, analysis_id)
);

SELECT create_hypertable('rca_analysis', 'time', chunk_time_interval => INTERVAL '1 day');

-- ML model predictions (hypertable)
CREATE TABLE predictions (
    time TIMESTAMP WITH TIME ZONE NOT NULL,
    prediction_id UUID DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    prediction_type VARCHAR(50) NOT NULL, -- FAILURE, MAINTENANCE, CAPACITY
    target_time TIMESTAMP WITH TIME ZONE NOT NULL,
    site_id UUID,
    tenant_id UUID,
    predicted_value JSONB NOT NULL,
    confidence_score DECIMAL(3,2),
    model_version VARCHAR(20),
    metadata JSONB,
    PRIMARY KEY (time, prediction_id)
);

SELECT create_hypertable('predictions', 'time', chunk_time_interval => INTERVAL '1 day');

-- =============================================================================
-- INTEGRATION TABLES
-- =============================================================================

-- ServiceNow tickets integration
CREATE TABLE itsm_tickets (
    ticket_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    external_ticket_id VARCHAR(100),
    ticket_type VARCHAR(50) NOT NULL, -- INCIDENT, REQUEST, CHANGE
    priority VARCHAR(20) NOT NULL,
    status VARCHAR(50) NOT NULL,
    title VARCHAR(500) NOT NULL,
    description TEXT,
    site_id UUID,
    tenant_id UUID,
    assigned_to VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    resolved_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB
);

-- Data sources configuration
CREATE TABLE data_sources (
    source_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_name VARCHAR(100) NOT NULL UNIQUE,
    source_type VARCHAR(50) NOT NULL, -- OSS, ITSM, IOT, ENERGY
    connection_config JSONB NOT NULL,
    mapping_rules JSONB,
    last_sync TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) DEFAULT 'active',
    tenant_id UUID,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =============================================================================
-- INDEXES FOR PERFORMANCE
-- =============================================================================

-- Sites indexes
CREATE INDEX idx_sites_tenant_id ON sites(tenant_id);
CREATE INDEX idx_sites_region ON sites(region);
CREATE INDEX idx_sites_technology ON sites USING GIN(technology);
CREATE INDEX idx_sites_location ON sites(latitude, longitude);

-- Network metrics indexes
CREATE INDEX idx_network_metrics_site_tenant ON network_metrics(site_id, tenant_id, time DESC);
CREATE INDEX idx_network_metrics_metric_name ON network_metrics(metric_name, time DESC);

-- Energy metrics indexes
CREATE INDEX idx_energy_metrics_site_type ON energy_metrics(site_id, energy_type, time DESC);

-- Events indexes
CREATE INDEX idx_events_site_tenant ON events(site_id, tenant_id, time DESC);
CREATE INDEX idx_events_severity ON events(severity, time DESC);
CREATE INDEX idx_events_type ON events(event_type, time DESC);
CREATE INDEX idx_events_resolved ON events(resolved, time DESC);

-- KPI values indexes
CREATE INDEX idx_kpi_values_kpi_site ON kpi_values(kpi_id, site_id, time DESC);
CREATE INDEX idx_kpi_values_tenant ON kpi_values(tenant_id, time DESC);

-- =============================================================================
-- FUNCTIONS AND PROCEDURES
-- =============================================================================

-- Function to calculate NUR (Network Unavailability Rate)
CREATE OR REPLACE FUNCTION calculate_nur(
    p_site_id UUID,
    p_start_time TIMESTAMP WITH TIME ZONE,
    p_end_time TIMESTAMP WITH TIME ZONE
)
RETURNS DECIMAL(5,4) AS $$
DECLARE
    total_minutes INTEGER;
    unavailable_minutes INTEGER;
BEGIN
    total_minutes := EXTRACT(EPOCH FROM (p_end_time - p_start_time)) / 60;
    
    SELECT COALESCE(SUM(
        CASE 
            WHEN metric_value < 0.95 THEN 5 -- 5-minute intervals
            ELSE 0 
        END
    ), 0) INTO unavailable_minutes
    FROM network_metrics
    WHERE site_id = p_site_id
    AND metric_name = 'availability_pct'
    AND time BETWEEN p_start_time AND p_end_time;
    
    RETURN ROUND((unavailable_minutes::DECIMAL / total_minutes) * 100, 4);
END;
$$ LANGUAGE plpgsql;

-- Function to get site health score
CREATE OR REPLACE FUNCTION get_site_health_score(p_site_id UUID)
RETURNS DECIMAL(3,2) AS $$
DECLARE
    network_score DECIMAL(3,2);
    energy_score DECIMAL(3,2);
    incident_score DECIMAL(3,2);
    overall_score DECIMAL(3,2);
BEGIN
    -- Network health (last hour average)
    SELECT COALESCE(AVG(quality_score), 0.5)
    INTO network_score
    FROM network_metrics
    WHERE site_id = p_site_id
    AND time >= NOW() - INTERVAL '1 hour';
    
    -- Energy health (last hour average)
    SELECT COALESCE(AVG(efficiency_score), 0.5)
    INTO energy_score
    FROM energy_metrics
    WHERE site_id = p_site_id
    AND time >= NOW() - INTERVAL '1 hour';
    
    -- Incident impact (last 24h)
    SELECT CASE
        WHEN COUNT(*) FILTER (WHERE severity IN ('CRITICAL', 'MAJOR')) > 0 THEN 0.2
        WHEN COUNT(*) FILTER (WHERE severity = 'MINOR') > 0 THEN 0.6
        WHEN COUNT(*) FILTER (WHERE severity = 'WARNING') > 0 THEN 0.8
        ELSE 1.0
    END INTO incident_score
    FROM events
    WHERE site_id = p_site_id
    AND time >= NOW() - INTERVAL '24 hours'
    AND resolved = FALSE;
    
    -- Weighted average
    overall_score := (network_score * 0.4 + energy_score * 0.3 + incident_score * 0.3);
    
    RETURN ROUND(overall_score, 2);
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- INITIAL DATA
-- =============================================================================

-- Insert default tenant
INSERT INTO tenants (tenant_id, tenant_name, tenant_code, contact_email, sla_targets) VALUES (
    '11111111-1111-1111-1111-111111111111',
    'Demo Telecom Operator',
    'DEMO',
    'admin@demo-telecom.com',
    '{"availability": 99.9, "mttr_hours": 2, "energy_efficiency": 85}'
);

-- Insert sample KPI definitions
INSERT INTO kpi_definitions (kpi_name, kpi_category, calculation_formula, unit, target_value, warning_threshold, critical_threshold) VALUES
('Network Availability', 'NETWORK', 'AVG(availability_pct)', '%', 99.9, 99.5, 99.0),
('Call Success Rate', 'NETWORK', 'AVG(csr_pct)', '%', 98.5, 95.0, 90.0),
('Data Throughput', 'NETWORK', 'AVG(throughput_mbps)', 'Mbps', 100.0, 50.0, 20.0),
('Energy Efficiency', 'ENERGY', 'AVG(pue_ratio)', 'ratio', 1.5, 2.0, 2.5),
('Battery Health', 'ENERGY', 'AVG(battery_health_pct)', '%', 90.0, 80.0, 70.0),
('MTTR', 'OPERATIONAL', 'AVG(resolution_time_minutes)', 'minutes', 120, 240, 480),
('Fuel Consumption', 'ENERGY', 'SUM(fuel_liters_per_day)', 'L/day', 50.0, 75.0, 100.0);

-- Create retention policies (keep detailed data for 30 days, hourly aggregates for 1 year)
SELECT add_retention_policy('network_metrics', INTERVAL '30 days');
SELECT add_retention_policy('energy_metrics', INTERVAL '30 days');
SELECT add_retention_policy('events', INTERVAL '90 days');
SELECT add_retention_policy('kpi_values', INTERVAL '365 days');

-- Create continuous aggregates for performance
CREATE MATERIALIZED VIEW network_metrics_hourly
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', time) AS bucket,
    site_id,
    tenant_id,
    technology,
    metric_name,
    AVG(metric_value) as avg_value,
    MAX(metric_value) as max_value,
    MIN(metric_value) as min_value,
    AVG(quality_score) as avg_quality_score
FROM network_metrics
GROUP BY bucket, site_id, tenant_id, technology, metric_name;

SELECT add_continuous_aggregate_policy('network_metrics_hourly',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');

COMMIT;