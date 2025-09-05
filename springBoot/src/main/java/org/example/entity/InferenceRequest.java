package org.example.entity;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;

import java.sql.Timestamp;
import java.time.LocalDateTime;

@Entity
@Table(name = "inference_request")
@Getter @Setter
public class InferenceRequest {

    @Id
    @Column(name = "request_id", length = 36)
    private String requestId;          // UUID

    @Column(name = "model_name", nullable = false, length = 100)
    private String modelName;

    @Column(name = "model_version", nullable = false, length = 50)
    private String modelVersion;

    @Column(name = "scaler_name", nullable = false, length = 100)
    private String scalerName;

    @Column(name = "scaler_version", nullable = false, length = 50)
    private String scalerVersion;

    @Column(name = "source_system", length = 100)
    private String sourceSystem;

    @Column(name = "measurement_ts")
    private LocalDateTime measurementTs;

    @Column(name = "features_json", columnDefinition = "json", nullable = false)
    private String featuresJson;       // JSON_ARRAY(8) 문자열

    @Column(name = "created_at", insertable = false, updatable = false)
    private Timestamp createdAt;
}
