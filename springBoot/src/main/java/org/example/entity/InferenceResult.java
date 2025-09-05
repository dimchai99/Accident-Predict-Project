package org.example.entity;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;

import java.sql.Timestamp;

@Entity
@Table(name = "inference_result")
@Getter @Setter
public class InferenceResult {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "result_id")
    private Long resultId;

    @Column(name = "request_id", nullable = false, length = 36)
    private String requestId;

    @Column(name = "model_name", nullable = false, length = 100)
    private String modelName;

    @Column(name = "model_version", nullable = false, length = 50)
    private String modelVersion;

    @Column(name = "reconstruction_mse", nullable = false)
    private Double reconstructionMse;

    @Column(name = "threshold_name", length = 100)
    private String thresholdName;

    @Column(name = "threshold_value")
    private Double thresholdValue;

    @Column(name = "is_anomaly", nullable = false)
    private Boolean isAnomaly;

    @Column(name = "detail_json", columnDefinition = "json")
    private String detailJson;

    @Column(name = "created_at", insertable = false, updatable = false)
    private Timestamp createdAt;
}
