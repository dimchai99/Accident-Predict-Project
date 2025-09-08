package org.example.domain;

import jakarta.persistence.*;
import java.io.Serializable;
import lombok.*;

/**
 * blade_benchmark 테이블 매핑 엔티티
 *
 * DDL 참고:
 * CREATE TABLE blade_benchmark (
 *   blade_benchmark_id INT AUTO_INCREMENT PRIMARY KEY,
 *   relative_timestamp FLOAT,
 *   cut_torque DOUBLE,
 *   cut_lag_error DOUBLE,
 *   cut_position DOUBLE,
 *   cut_speed DOUBLE,
 *   film_position DOUBLE,
 *   film_speed DOUBLE,
 *   film_lag_error DOUBLE
 * );
 */
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
@ToString
@EqualsAndHashCode(of = "bladeBenchmarkId")
@Entity
@Table(name = "blade_benchmark")
public class BladeBenchmark implements Serializable {

    private static final long serialVersionUID = 1L;

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "blade_benchmark_id")
    private String bladeBenchmarkId;

    @Column(name = "relative_timestamp")
    private Double relativeTimestamp;

    @Column(name = "cut_torque")
    private Double cutTorque;

    @Column(name = "cut_lag_error")
    private Double cutLagError;

    @Column(name = "cut_position")
    private Double cutPosition;

    @Column(name = "cut_speed")
    private Double cutSpeed;

    @Column(name = "film_position")
    private Double filmPosition;

    @Column(name = "film_speed")
    private Double filmSpeed;

    @Column(name = "film_lag_error")
    private Double filmLagError;
}