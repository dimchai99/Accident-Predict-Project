package org.example.dto;

public record CreateRequestDto(
        String requestId,
        String modelName,
        String modelVersion,
        String scalerName,
        String scalerVersion,
        String sourceSystem,
        String measurementTs,   // ISO-8601 문자열
        String featuresJson     // JSON 문자열(길이 8 배열)
) {}
