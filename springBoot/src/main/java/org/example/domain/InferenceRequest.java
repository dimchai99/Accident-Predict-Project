package org.example.domain;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;
import org.springframework.data.jpa.repository.JpaRepository;

// org/example/domain/InferenceRequest.java
@Entity
@Table(name="inference_request")
@Getter
@Setter
public class InferenceRequest {
    @Id
    @GeneratedValue(strategy= GenerationType.IDENTITY)
    private Long requestId;
    private Integer assetId;
    private Integer sensorId;
    private java.sql.Timestamp tsFrom;
    private java.sql.Timestamp tsTo;
    private String status;  // PENDING, IN_PROGRESS, COMPLETED, FAILED
    private String message;
    private java.sql.Timestamp createdAt;
    private java.sql.Timestamp updatedAt;
}
