package org.example.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import jakarta.transaction.Transactional;
import lombok.RequiredArgsConstructor;
import org.example.client.PythonModelClient;
import org.example.repository.InferenceRequestRepo;
import org.example.domain.InferenceResult;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;

// org/example/service/InferenceOrchestrator.java
@Service
@RequiredArgsConstructor
public class InferenceOrchestrator {
    private final JdbcTemplate jdbc;                 // run_measurement 쿼리용
    private final InferenceRequestRepo reqRepo;
    private InferenceResult resRepo;
    private final PythonModelClient py;

    private static final java.util.List<String> ORDER = java.util.List.of(
            "relative_timestamp","cut_torque","cut_lag_error","cut_position",
            "cut_speed","film_position","film_speed","film_lag_error"
    );

    @Transactional
    public InferenceResult run(Long requestId) throws JsonProcessingException {
        var req = reqRepo.findById(requestId)
                .orElseThrow(() -> new IllegalArgumentException("request not found: " + requestId));
        req.setStatus("IN_PROGRESS"); reqRepo.save(req);

        // run_measurement에서 구간 데이터 가져오기
        var rows = jdbc.query("""
      SELECT CONCAT(sensor_id,'_',DATE_FORMAT(measurement_ts,'%Y%m%d%H%i%S')) AS id,
             relative_timestamp, cut_torque, cut_lag_error, cut_position,
             cut_speed, film_position, film_speed, film_lag_error
      FROM run_measurement
      WHERE sensor_id = ? AND measurement_ts BETWEEN ? AND ?
      ORDER BY measurement_ts ASC
    """, ps -> {
            ps.setInt(1, req.getSensorId());
            ps.setTimestamp(2, req.getTsFrom());
            ps.setTimestamp(3, req.getTsTo());
        }, (rs, i) -> new PythonModelClient.Row(
                rs.getString("id"),
                rs.getDouble("relative_timestamp"),
                rs.getDouble("cut_torque"),
                rs.getDouble("cut_lag_error"),
                rs.getDouble("cut_position"),
                rs.getDouble("cut_speed"),
                rs.getDouble("film_position"),
                rs.getDouble("film_speed"),
                rs.getDouble("film_lag_error")
        ));

        if (rows.isEmpty()) {
            req.setStatus("FAILED"); req.setMessage("no data"); reqRepo.save(req);
            throw new IllegalStateException("no run_measurement rows");
        }

        var resp = py.infer(rows, ORDER);

        // 집계(예: 평균 health)
        var avgHealth = resp.results().stream().mapToDouble(PythonModelClient.Item::health).average().orElse(Double.NaN);

        var detailsJson = new com.fasterxml.jackson.databind.ObjectMapper().writeValueAsString(resp);
        var out = new InferenceResult();
        out.setRequestId(requestId);
        out.setPredictedHealthScore(avgHealth);
        out.setPredictedRul(Math.max(0, (1.0 - avgHealth) * 100.0)); // 간단 RUL
        out.setDetailsJson(detailsJson);
        out.setCreatedAt(new java.sql.Timestamp(System.currentTimeMillis()));
        resRepo.save(out);

        req.setStatus("COMPLETED"); req.setMessage("OK"); reqRepo.save(req);
        return out;
    }
}
