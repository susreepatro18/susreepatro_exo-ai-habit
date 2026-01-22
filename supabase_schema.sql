-- Create exoplanets table for storing exoplanet data
CREATE TABLE IF NOT EXISTS exoplanets (
    id SERIAL PRIMARY KEY,
    pl_name VARCHAR(255) UNIQUE NOT NULL,
    hostname VARCHAR(255),
    pl_type VARCHAR(100),
    pl_bmassj DECIMAL(10, 6),
    pl_radj DECIMAL(10, 6),
    pl_orbper DECIMAL(15, 6),
    pl_orbsmax DECIMAL(15, 8),
    st_teff INTEGER,
    st_rad DECIMAL(10, 6),
    st_mass DECIMAL(10, 6),
    sy_dist DECIMAL(10, 6),
    disc_year INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create predictions table for storing model predictions
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    exoplanet_id INTEGER REFERENCES exoplanets(id) ON DELETE CASCADE,
    pl_name VARCHAR(255),
    prediction_type VARCHAR(50),
    prediction_value VARCHAR(100),
    confidence_score DECIMAL(5, 4),
    model_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create rankings table for storing user rankings
CREATE TABLE IF NOT EXISTS rankings (
    id SERIAL PRIMARY KEY,
    exoplanet_id INTEGER REFERENCES exoplanets(id) ON DELETE CASCADE,
    pl_name VARCHAR(255),
    rank_score DECIMAL(5, 2),
    rank_category VARCHAR(100),
    habitability_score DECIMAL(5, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better query performance
CREATE INDEX idx_exoplanets_name ON exoplanets(pl_name);
CREATE INDEX idx_predictions_exoplanet ON predictions(exoplanet_id);
CREATE INDEX idx_predictions_type ON predictions(prediction_type);
CREATE INDEX idx_rankings_exoplanet ON rankings(exoplanet_id);
CREATE INDEX idx_rankings_score ON rankings(rank_score);

-- Enable RLS (Row Level Security) for security
ALTER TABLE exoplanets ENABLE ROW LEVEL SECURITY;
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE rankings ENABLE ROW LEVEL SECURITY;

-- Create policies to allow public read access
CREATE POLICY "Allow public read on exoplanets" ON exoplanets
    FOR SELECT USING (true);

CREATE POLICY "Allow public read on predictions" ON predictions
    FOR SELECT USING (true);

CREATE POLICY "Allow public read on rankings" ON rankings
    FOR SELECT USING (true);

-- Create policies to allow insertions for API operations
CREATE POLICY "Allow insert on exoplanets" ON exoplanets
    FOR INSERT WITH CHECK (true);

CREATE POLICY "Allow insert on predictions" ON predictions
    FOR INSERT WITH CHECK (true);

CREATE POLICY "Allow insert on rankings" ON rankings
    FOR INSERT WITH CHECK (true);
