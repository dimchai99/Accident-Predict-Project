// org/example/controller/InferenceController.java

package org.example.controller;


import com.fasterxml.jackson.core.JsonProcessingException;
import lombok.RequiredArgsConstructor;
import org.example.domain.InferenceResult;
import org.example.service.InferenceOrchestrator;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/infer")
public class InferenceController {
    private final InferenceOrchestrator orchestrator;

    @PostMapping("/request/{id}/run")
    public InferenceResult runByRequest(@PathVariable Long id) throws JsonProcessingException {
        return orchestrator.run(id);
    }
}
