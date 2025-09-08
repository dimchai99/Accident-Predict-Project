package org.example.repository;

import org.example.domain.BenchmarkScore;
import org.springframework.data.jpa.repository.JpaRepository;

public interface BenchmarkScoreRepo extends JpaRepository<BenchmarkScore, String> {
}
