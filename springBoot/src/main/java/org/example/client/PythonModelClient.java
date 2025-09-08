package org.example.client;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

@Component
@RequiredArgsConstructor
public class PythonModelClient {

    @Value("${app.python.exec}")   private String pythonExec;
    @Value("${app.python.script}") private String scriptPath;
    @Value("${app.python.scaler}") private String scalerPath;
    @Value("${app.python.model}")  private String modelPath;

    private final ObjectMapper om;

    // 입력 1행 DTO (blade_benchmark 레코드)
    public record Row(
            String id,
            Double relative_timestamp, Double cut_torque, Double cut_lag_error, Double cut_position,
            Double cut_speed, Double film_position, Double film_speed, Double film_lag_error
    ) {}

    public record Req(java.util.List<Row> rows, java.util.List<String> feature_order) {}
    public record Item(String id, Double health, Double mse) {}
    public record Resp(java.util.List<Item> results) {}

    public Resp infer(java.util.List<Row> rows, java.util.List<String> order) {
        try {
            var pb = new ProcessBuilder(
                    pythonExec, scriptPath, "--scaler", scalerPath, "--model", modelPath
            );
            pb.redirectErrorStream(true);
            var proc = pb.start();

            // stdin으로 JSON 쓰기
            try (var out = proc.getOutputStream()) {
                om.writeValue(out, new Req(rows, order));
                out.flush();
            }

            // stdout에서 JSON 읽기
            Resp resp;
            try (var in = proc.getInputStream()) {
                resp = om.readValue(in, Resp.class);
            }

            int code = proc.waitFor();
            if (code != 0) throw new RuntimeException("Python process exited " + code);
            return resp;
        } catch (Exception e) {
            throw new RuntimeException("infer failed: " + e.getMessage(), e);
        }
    }
}
