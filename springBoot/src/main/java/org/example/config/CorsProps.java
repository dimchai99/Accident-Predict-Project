package org.example.config;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;

import java.util.ArrayList;
import java.util.List;

@Data  // ➝ getter/setter, toString, equals/hashCode, 기본 생성자까지 자동 생성
@ConfigurationProperties(prefix = "app.cors")
public class CorsProps {
    private List<String> origins = new ArrayList<>();
    private Boolean allowCredentials = true;
    private List<String> allowedMethods = List.of("GET","POST","PUT","PATCH","DELETE","OPTIONS");
    private List<String> allowedHeaders = List.of("*");
    private List<String> exposedHeaders = List.of("Content-Disposition");
}