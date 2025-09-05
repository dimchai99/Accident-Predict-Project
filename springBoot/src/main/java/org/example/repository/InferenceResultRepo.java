package org.example.repository;

import org.example.entity.InferenceResult;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface InferenceResultRepo extends JpaRepository<InferenceResult, Long> {
    List<InferenceResult> findByRequestId(String requestId);
}
