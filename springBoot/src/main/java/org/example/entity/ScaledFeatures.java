package org.example.entity;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;

import java.sql.Timestamp;

@Entity
@Table(name = "scaled_features")
@Getter @Setter
public class ScaledFeatures {

    @Id
    @Column(name = "request_id", length = 36)
    private String requestId;   // FK to inference_request.request_id

    @Column(name = "scaled_json", columnDefinition = "json", nullable = false)
    private String scaledJson;  // JSON_ARRAY(8)

    @Column(name = "created_at", insertable = false, updatable = false)
    private Timestamp createdAt;
}
