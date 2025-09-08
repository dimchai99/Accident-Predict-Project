package org.example.domain;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;

// org/example/domain/InferenceResult.java
@Entity @Table(name="inference_result")
@Getter @Setter
public class InferenceResult {
    @Id @GeneratedValue(strategy= GenerationType.IDENTITY)
    private Long resultId;
    private Long requestId;
    private Double predictedHealthScore;
    private Double predictedRul;
    @Lob private String detailsJson;
    private java.sql.Timestamp createdAt;

    public void save(InferenceResult out) {

    }
}
