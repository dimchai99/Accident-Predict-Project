package org.example.dto;

public record SaveScaledDto(
        String requestId,
        String scaledJson       // JSON 문자열(길이 8 배열)
) {}
