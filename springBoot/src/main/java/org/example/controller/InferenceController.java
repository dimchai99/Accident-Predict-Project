package org.example.controller;

import org.example.dto.CreateRequestDto;
import org.example.dto.SaveResultDto;
import org.example.dto.SaveScaledDto;
import org.example.entity.InferenceRequest;
import org.example.entity.InferenceResult;
import org.example.entity.ScaledFeatures;
import org.example.repository.InferenceRequestRepo;
import org.example.repository.InferenceResultRepo;
import org.example.repository.ScaledFeaturesRepo;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.sql.Timestamp;
import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/api/infer")
@RequiredArgsConstructor
public class InferenceController {

    private final InferenceRequestRepo requestRepo;
    private final ScaledFeaturesRepo scaledRepo;
    private final InferenceResultRepo resultRepo;

    // (옵션) 요청 생성: 파이썬이 직접 DB 넣지 않고 HTTP로 생성하고 싶을 때 사용
    @PostMapping("/requests")
    public Map<String, Object> createRequest(@RequestBody CreateRequestDto dto) {
        var req = new InferenceRequest();
        var id = (dto.requestId() == null || dto.requestId().isBlank())
                ? java.util.UUID.randomUUID().toString() : dto.requestId();

        req.setRequestId(id);
        req.setModelName(dto.modelName());
        req.setModelVersion(dto.modelVersion());
        req.setScalerName(dto.scalerName());
        req.setScalerVersion(dto.scalerVersion());
        req.setSourceSystem(dto.sourceSystem());
        if (dto.measurementTs() != null && !dto.measurementTs().isBlank()) {
            req.setMeasurementTs(LocalDateTime.parse(dto.measurementTs()));
        }
        req.setFeaturesJson(dto.featuresJson()); // 길이 8 검증은 서비스단/필요시 추가
        requestRepo.save(req);

        return Map.of("requestId", id);
    }

    // 파이썬: /api/infer/scaled 로 스케일 값 저장
    @PostMapping("/scaled")
    public Map<String, Object> saveScaled(@RequestBody SaveScaledDto dto) {
        var s = new ScaledFeatures();
        s.setRequestId(dto.requestId());
        s.setScaledJson(dto.scaledJson());
        scaledRepo.save(s);
        return Map.of("saved", true);
    }

    // 파이썬: /api/infer/results 로 결과 저장
    @PostMapping("/results")
    public Map<String, Object> saveResult(@RequestBody SaveResultDto dto) {
        var r = new InferenceResult();
        r.setRequestId(dto.requestId());
        r.setModelName(dto.modelName());
        r.setModelVersion(dto.modelVersion());
        r.setReconstructionMse(dto.reconstructionMse());
        r.setThresholdName(dto.thresholdName());
        r.setThresholdValue(dto.thresholdValue());
        r.setIsAnomaly(Boolean.TRUE.equals(dto.isAnomaly()));
        r.setDetailJson(dto.detailJson());
        resultRepo.save(r);

        var resp = new HashMap<String, Object>();
        resp.put("resultId", r.getResultId());
        resp.put("createdAt", r.getCreatedAt() != null ? r.getCreatedAt() : Timestamp.from(java.time.Instant.now()));
        return resp;
    }

    // (편의) 요청/스케일/결과 한 큐에 보기
    @GetMapping("/{requestId}")
    public Map<String, Object> view(@PathVariable String requestId) {
        var req = requestRepo.findById(requestId).orElse(null);
        var scaled = scaledRepo.findById(requestId).orElse(null);
        var results = resultRepo.findByRequestId(requestId);
        return Map.of("request", req, "scaled", scaled, "results", results);
    }
}
