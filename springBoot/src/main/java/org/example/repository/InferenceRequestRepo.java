package org.example.repository;

import org.example.entity.InferenceRequest;
import org.springframework.data.jpa.repository.JpaRepository;

public interface InferenceRequestRepo extends JpaRepository<InferenceRequest, String> {}
