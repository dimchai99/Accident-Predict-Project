package org.example.repository;

import org.example.domain.InferenceRequest;
import org.springframework.data.jpa.repository.JpaRepository;

// repo
public interface InferenceRequestRepo extends JpaRepository<InferenceRequest, Long> {}
