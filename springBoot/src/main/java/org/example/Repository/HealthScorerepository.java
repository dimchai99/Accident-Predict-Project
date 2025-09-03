package org.example.Repository;

import org.example.entity.HealthScore;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.sql.Timestamp;
import java.util.List;

public interface HealthScorerepository extends JpaRepository<HealthScore, Long> {

    @Query("""
    SELECT h FROM HealthScore h
    WHERE (:assetId IS NULL OR h.assetId = :assetId)
      AND h.ts BETWEEN :from AND :to
    ORDER BY h.ts
  """)
    List<HealthScore> findSeries(
            @Param("assetId") Integer assetId,
            @Param("from") Timestamp from,
            @Param("to")   Timestamp to
    );
}
