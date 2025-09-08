package org.example.repository;

import org.example.domain.BladeBenchmark;
import org.springframework.data.jpa.repository.JpaRepository;

public interface BladeBenchmarkRepo extends JpaRepository<BladeBenchmark, String> {
}
