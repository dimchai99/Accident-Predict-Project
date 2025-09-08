package org.example.repository;

import jakarta.persistence.*;
import lombok.*;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
@ToString
@EqualsAndHashCode(of = "bladeInboundId")
@Entity
@Table(name = "benchmark_score")
public class BenchmarkScore {

    @Id
    @Column(name = "blade_inbound_id")
    private String bladeInboundId;   // VARCHAR(3), blade_benchmark PK 참조

    @Column(name = "health_score")
    private Double healthScore;
}