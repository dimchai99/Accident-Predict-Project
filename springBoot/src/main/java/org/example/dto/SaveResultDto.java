package org.example.dto;

public record SaveResultDto(
        String requestId,
        String modelName,
        String modelVersion,
        Double reconstructionMse,
        String thresholdName,
        Double thresholdValue,
        Boolean isAnomaly,
        String detailJson       // JSON 문자열(예: per_feature_sqerr)
) {}
