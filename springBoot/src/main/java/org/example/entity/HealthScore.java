package org.example.entity;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;

import java.sql.Timestamp;

// HealthScore.java
@Entity
@Table(name="health_score")
@Getter
@Setter
public class HealthScore {
    @Id
    @GeneratedValue(strategy= GenerationType.IDENTITY)
    private Long healthScoreId;

    private Integer assetId;
    private Integer componentId;
    private Integer bladeReplacementId;

    private Timestamp ts;

    private Double score;
    private Double rulCycles;

    private String state;
    private String method;

    @Column(columnDefinition="JSON")
    private String params;

    private String source;

    private Long runId;
}
