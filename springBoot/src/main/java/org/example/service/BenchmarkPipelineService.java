package org.example.service;

import org.example.client.PythonModelClient;
import org.example.domain.BenchmarkScore;
import org.example.repository.BladeBenchmarkRepo;
import org.example.repository.BenchmarkScoreRepo;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service @RequiredArgsConstructor
public class BenchmarkPipelineService {

    private final BladeBenchmarkRepo benchmarkRepo;
    private final BenchmarkScoreRepo scoreRepo;
    private final PythonModelClient py;

    // 스케일러 학습 시 사용한 컬럼 순서 (파이썬 infer_cli와 동일해야 함)
    private static final java.util.List<String> ORDER = java.util.List.of(
            "relative_timestamp","cut_torque","cut_lag_error","cut_position",
            "cut_speed","film_position","film_speed","film_lag_error"
    );

    @Transactional
    public java.util.List<BenchmarkScore> runAndSaveAll() {
        var rows = benchmarkRepo.findAll().stream().map(b ->
                new PythonModelClient.Row(
                        b.getBladeBenchmarkId(),
                        b.getRelativeTimestamp(), b.getCutTorque(), b.getCutLagError(), b.getCutPosition(),
                        b.getCutSpeed(), b.getFilmPosition(), b.getFilmSpeed(), b.getFilmLagError()
                )
        ).toList();

        var resp = py.infer(rows, ORDER);

        var out = new java.util.ArrayList<BenchmarkScore>();
        for (var it : resp.results()) {
            var e = new BenchmarkScore();
            e.setBladeInboundId(it.id());
            e.setHealthScore(it.health());
            // e.setMse(it.mse());
            out.add(e);
        }
        return scoreRepo.saveAll(out);
    }
}
