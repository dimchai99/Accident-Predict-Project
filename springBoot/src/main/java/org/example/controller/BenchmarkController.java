package org.example.controller;

import org.example.domain.BenchmarkScore;
import org.example.service.BenchmarkPipelineService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/benchmark")
public class BenchmarkController {

    private final BenchmarkPipelineService svc;

    @PostMapping("/score/refresh")
    public java.util.List<BenchmarkScore> refreshScores() {
        return svc.runAndSaveAll();
    }
}
