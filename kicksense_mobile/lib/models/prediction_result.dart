import 'dart:convert';
import 'dart:typed_data';

class BiomechanicsData {
  final double torsoLeanDeg;
  final String torsoLeanLabel;
  final double pelvisAngleDeg;
  final String pelvisAngleLabel;
  final double plantToBallDist;
  final double kickingAnklePeakSpeed;

  BiomechanicsData({
    required this.torsoLeanDeg,
    required this.torsoLeanLabel,
    required this.pelvisAngleDeg,
    required this.pelvisAngleLabel,
    required this.plantToBallDist,
    required this.kickingAnklePeakSpeed,
  });

  factory BiomechanicsData.fromJson(Map<String, dynamic> json) {
    return BiomechanicsData(
      torsoLeanDeg: (json['torso_lean_deg'] as num?)?.toDouble() ?? 0.0,
      torsoLeanLabel: json['torso_lean_label'] as String? ?? 'Normal Posture',
      pelvisAngleDeg: (json['pelvis_angle_deg'] as num?)?.toDouble() ?? 0.0,
      pelvisAngleLabel: json['pelvis_angle_label'] as String? ?? 'Squared Hips',
      plantToBallDist: (json['plant_to_ball_dist'] as num?)?.toDouble() ?? 0.0,
      kickingAnklePeakSpeed: (json['kicking_ankle_peak_speed'] as num?)?.toDouble() ?? 0.0,
    );
  }
}

class PredictionResult {
  final String direction;
  final double confidence;
  final Map<String, double> probabilities;
  final double power;
  final String powerCategory;
  final int strikeFrameIdx;
  final int startFrameIdx;
  final int totalFrames;
  final double fps;
  final String? strikeSkeletonBase64;
  final BiomechanicsData? biomechanics;
  final double inferenceTimeMs;

  PredictionResult({
    required this.direction,
    required this.confidence,
    required this.probabilities,
    required this.power,
    required this.powerCategory,
    required this.strikeFrameIdx,
    required this.startFrameIdx,
    required this.totalFrames,
    required this.fps,
    this.strikeSkeletonBase64,
    this.biomechanics,
    required this.inferenceTimeMs,
  });

  Uint8List? get skeletonBytes {
    if (strikeSkeletonBase64 == null || strikeSkeletonBase64!.isEmpty) {
      return null;
    }
    try {
      return base64Decode(strikeSkeletonBase64!);
    } catch (_) {
      return null;
    }
  }

  factory PredictionResult.fromJson(Map<String, dynamic> json) {
    final probsRaw = json['probabilities'] as Map<String, dynamic>? ?? {};
    final probs = <String, double>{
      'Left': (probsRaw['Left'] as num?)?.toDouble() ?? 0.0,
      'Center': (probsRaw['Center'] as num?)?.toDouble() ?? 0.0,
      'Right': (probsRaw['Right'] as num?)?.toDouble() ?? 0.0,
    };

    return PredictionResult(
      direction: json['direction'] as String? ?? 'Center',
      confidence: (json['confidence'] as num?)?.toDouble() ?? 0.0,
      probabilities: probs,
      power: (json['power'] as num?)?.toDouble() ?? 5.0,
      powerCategory: json['power_category'] as String? ?? 'Moderate Power',
      strikeFrameIdx: (json['strike_frame_idx'] as num?)?.toInt() ?? 0,
      startFrameIdx: (json['start_frame_idx'] as num?)?.toInt() ?? 0,
      totalFrames: (json['total_frames'] as num?)?.toInt() ?? 0,
      fps: (json['fps'] as num?)?.toDouble() ?? 30.0,
      strikeSkeletonBase64: json['strike_skeleton_base64'] as String?,
      biomechanics: json['biomechanics'] != null
          ? BiomechanicsData.fromJson(json['biomechanics'] as Map<String, dynamic>)
          : null,
      inferenceTimeMs: (json['inference_time_ms'] as num?)?.toDouble() ?? 0.0,
    );
  }
}
