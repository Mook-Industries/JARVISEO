-- scripts/schema_sql.py 가 memory/models.py 에서 뽑은 것. 손으로 고치지 않는다.
-- 없는 테이블 · 인덱스만 만든다. 열을 바꾼 것은 반영되지 않는다(init_schema 와 같다).
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS allergen (
	allergen_id SMALLSERIAL NOT NULL, 
	name VARCHAR(50) NOT NULL, 
	category VARCHAR(20) NOT NULL, 
	PRIMARY KEY (allergen_id), 
	UNIQUE (name)
);

CREATE TABLE IF NOT EXISTS eval_run (
	run_id BIGSERIAL NOT NULL, 
	variant VARCHAR(20) NOT NULL, 
	run_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	sample_count INTEGER NOT NULL, 
	metrics JSON NOT NULL, 
	note TEXT, 
	PRIMARY KEY (run_id)
);

CREATE INDEX IF NOT EXISTS ix_eval_run_variant ON eval_run (variant);

CREATE TABLE IF NOT EXISTS product (
	barcode VARCHAR(20) NOT NULL, 
	product_name VARCHAR(200) NOT NULL, 
	report_no VARCHAR(30) NOT NULL, 
	raw_ingredients TEXT NOT NULL, 
	ingredients JSON NOT NULL, 
	allergen_notice VARCHAR(500), 
	cross_contamination VARCHAR(500), 
	status VARCHAR(10), 
	verify_count SMALLINT, 
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	PRIMARY KEY (barcode)
);

CREATE TABLE IF NOT EXISTS users (
	user_id BIGSERIAL NOT NULL, 
	email VARCHAR(255) NOT NULL, 
	password_hash VARCHAR(60) NOT NULL, 
	nickname VARCHAR(50), 
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	PRIMARY KEY (user_id), 
	UNIQUE (email)
);

CREATE TABLE IF NOT EXISTS chat_session (
	session_id BIGSERIAL NOT NULL, 
	user_id BIGINT, 
	title VARCHAR(200), 
	started_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	ended_at TIMESTAMP WITH TIME ZONE, 
	end_reason VARCHAR(20), 
	PRIMARY KEY (session_id), 
	FOREIGN KEY(user_id) REFERENCES users (user_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS ix_chat_session_user_id ON chat_session (user_id);

CREATE TABLE IF NOT EXISTS ingredient_synonym (
	synonym_id BIGSERIAL NOT NULL, 
	allergen_id SMALLINT NOT NULL, 
	alias VARCHAR(100) NOT NULL, 
	PRIMARY KEY (synonym_id), 
	FOREIGN KEY(allergen_id) REFERENCES allergen (allergen_id) ON DELETE CASCADE, 
	UNIQUE (alias)
);

CREATE INDEX IF NOT EXISTS ix_ingredient_synonym_allergen_id ON ingredient_synonym (allergen_id);

CREATE TABLE IF NOT EXISTS user_allergen (
	user_allergen_id BIGSERIAL NOT NULL, 
	user_id BIGINT NOT NULL, 
	allergen_id SMALLINT NOT NULL, 
	is_strict BOOLEAN NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	deleted_at TIMESTAMP WITH TIME ZONE, 
	PRIMARY KEY (user_allergen_id), 
	FOREIGN KEY(user_id) REFERENCES users (user_id) ON DELETE CASCADE, 
	FOREIGN KEY(allergen_id) REFERENCES allergen (allergen_id)
);

CREATE INDEX IF NOT EXISTS ix_user_allergen_allergen_id ON user_allergen (allergen_id);

CREATE INDEX IF NOT EXISTS ix_user_allergen_user_id ON user_allergen (user_id);

CREATE UNIQUE INDEX IF NOT EXISTS uq_user_allergen_alive ON user_allergen (user_id, allergen_id) WHERE deleted_at IS NULL;

CREATE TABLE IF NOT EXISTS user_setting (
	user_id BIGINT NOT NULL, 
	tts_volume SMALLINT NOT NULL, 
	tts_speed VARCHAR(10) NOT NULL, 
	tts_voice VARCHAR(50) NOT NULL, 
	session_timeout_min SMALLINT NOT NULL, 
	show_detection_box BOOLEAN NOT NULL, 
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	PRIMARY KEY (user_id), 
	FOREIGN KEY(user_id) REFERENCES users (user_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS session_turn (
	turn_id BIGSERIAL NOT NULL, 
	session_id BIGINT NOT NULL, 
	asked_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	trigger_type VARCHAR(20) NOT NULL, 
	question_text TEXT NOT NULL, 
	answer_text TEXT, 
	image_path VARCHAR(500), 
	ocr_text TEXT, 
	total_ms INTEGER, 
	PRIMARY KEY (turn_id), 
	FOREIGN KEY(session_id) REFERENCES chat_session (session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS ix_session_turn_asked_at ON session_turn (asked_at);

CREATE INDEX IF NOT EXISTS ix_session_turn_session_id ON session_turn (session_id);

CREATE TABLE IF NOT EXISTS eval_sample (
	sample_id BIGSERIAL NOT NULL, 
	run_id BIGINT NOT NULL, 
	turn_id BIGINT, 
	gt_label VARCHAR(50) NOT NULL, 
	pred_label VARCHAR(50), 
	is_correct BOOLEAN NOT NULL, 
	PRIMARY KEY (sample_id), 
	FOREIGN KEY(run_id) REFERENCES eval_run (run_id) ON DELETE CASCADE, 
	FOREIGN KEY(turn_id) REFERENCES session_turn (turn_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS ix_eval_sample_is_correct ON eval_sample (is_correct);

CREATE INDEX IF NOT EXISTS ix_eval_sample_run_id ON eval_sample (run_id);

CREATE INDEX IF NOT EXISTS ix_eval_sample_turn_id ON eval_sample (turn_id);

CREATE TABLE IF NOT EXISTS observation (
	observation_id BIGSERIAL NOT NULL, 
	user_id BIGINT NOT NULL, 
	turn_id BIGINT NOT NULL, 
	description TEXT, 
	place VARCHAR(100), 
	embedding VECTOR(1536), 
	embed_model VARCHAR(50), 
	observed_at TIMESTAMP WITH TIME ZONE DEFAULT now(), 
	PRIMARY KEY (observation_id), 
	FOREIGN KEY(user_id) REFERENCES users (user_id) ON DELETE CASCADE, 
	FOREIGN KEY(turn_id) REFERENCES session_turn (turn_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS ix_observation_turn_id ON observation (turn_id);

CREATE INDEX IF NOT EXISTS ix_observation_user_id ON observation (user_id);

CREATE TABLE IF NOT EXISTS turn_candidate (
	candidate_id BIGSERIAL NOT NULL, 
	turn_id BIGINT NOT NULL, 
	rank SMALLINT NOT NULL, 
	label VARCHAR(50) NOT NULL, 
	score NUMERIC(4, 3) NOT NULL, 
	bbox JSON NOT NULL, 
	cue_scores JSON NOT NULL, 
	is_chosen BOOLEAN NOT NULL, 
	PRIMARY KEY (candidate_id), 
	FOREIGN KEY(turn_id) REFERENCES session_turn (turn_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS ix_turn_candidate_label ON turn_candidate (label);

CREATE INDEX IF NOT EXISTS ix_turn_candidate_turn_id ON turn_candidate (turn_id);

CREATE UNIQUE INDEX IF NOT EXISTS uq_turn_candidate_chosen ON turn_candidate (turn_id) WHERE is_chosen;

CREATE TABLE IF NOT EXISTS turn_inference (
	turn_id BIGINT NOT NULL, 
	target_label VARCHAR(50), 
	confidence NUMERIC(4, 3), 
	margin NUMERIC(4, 3), 
	pointing_variant VARCHAR(20), 
	fingertip_xy JSON, 
	blur_score NUMERIC(8, 2), 
	resolved_by VARCHAR(10), 
	is_reask BOOLEAN NOT NULL, 
	is_retake BOOLEAN NOT NULL, 
	vision_ms INTEGER, 
	llm_ms INTEGER, 
	route_ms INTEGER, 
	chosen_candidate_id BIGINT, 
	frame_offset_ms INTEGER, 
	detected_count SMALLINT, 
	detector_version VARCHAR(30), 
	PRIMARY KEY (turn_id), 
	FOREIGN KEY(turn_id) REFERENCES session_turn (turn_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS ix_turn_inference_pointing_variant ON turn_inference (pointing_variant);

CREATE TABLE IF NOT EXISTS turn_ingredient (
	turn_id BIGINT NOT NULL, 
	barcode VARCHAR(20), 
	source VARCHAR(10), 
	verdict VARCHAR(15), 
	matched JSON NOT NULL, 
	may_contain JSON NOT NULL, 
	normalized JSON NOT NULL, 
	ocr_conf NUMERIC(4, 3), 
	retry_count SMALLINT NOT NULL, 
	response_text TEXT, 
	ocr_ms INTEGER, 
	match_ms INTEGER, 
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	PRIMARY KEY (turn_id), 
	FOREIGN KEY(turn_id) REFERENCES session_turn (turn_id) ON DELETE CASCADE, 
	FOREIGN KEY(barcode) REFERENCES product (barcode)
);

CREATE INDEX IF NOT EXISTS ix_turn_ingredient_barcode ON turn_ingredient (barcode);

CREATE INDEX IF NOT EXISTS ix_turn_ingredient_verdict ON turn_ingredient (verdict);

CREATE TABLE IF NOT EXISTS turn_voice (
	turn_id BIGINT NOT NULL, 
	speech_started_at TIMESTAMP WITH TIME ZONE, 
	speech_ended_at TIMESTAMP WITH TIME ZONE, 
	stt_raw_text TEXT, 
	is_interrupted BOOLEAN NOT NULL, 
	is_filler_sent BOOLEAN NOT NULL, 
	audio_path VARCHAR(500), 
	stt_ms INTEGER, 
	tts_ms INTEGER, 
	PRIMARY KEY (turn_id), 
	FOREIGN KEY(turn_id) REFERENCES session_turn (turn_id) ON DELETE CASCADE
);
