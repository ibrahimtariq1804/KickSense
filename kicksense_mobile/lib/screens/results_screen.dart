import 'dart:io';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../models/prediction_result.dart';
import '../theme/app_theme.dart';
import '../widgets/goal_visualizer.dart';
import '../widgets/power_meter.dart';
import '../widgets/telemetry_card.dart';
import '../widgets/video_replay_card.dart';

class ResultsScreen extends StatelessWidget {
  final PredictionResult result;
  final File? videoFile;
  final String? videoUrl;
  final String? sampleName;
  final String? trueDirection;
  final double? truePower;

  const ResultsScreen({
    super.key,
    required this.result,
    this.videoFile,
    this.videoUrl,
    this.sampleName,
    this.trueDirection,
    this.truePower,
  });

  @override
  Widget build(BuildContext context) {
    String arrow = '⚽';
    if (result.direction == 'Left') arrow = '⬅️';
    if (result.direction == 'Center') arrow = '⬆️';
    if (result.direction == 'Right') arrow = '➡️';

    final bool hasTrueLabel = trueDirection != null;
    final bool isCorrect = hasTrueLabel && trueDirection!.toLowerCase() == result.direction.toLowerCase();

    return Scaffold(
      appBar: AppBar(
        title: const Text('Analysis Results'),
        actions: [
          IconButton(
            icon: const Icon(Icons.share_outlined),
            onPressed: () {
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text('Result summary copied to clipboard!')),
              );
            },
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Hero Direction Banner
            Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                gradient: const LinearGradient(
                  colors: [Color(0xFF0F2644), Color(0xFF0D3B3F)],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: AppTheme.accentEmerald.withValues(alpha: 0.4)),
                boxShadow: [
                  BoxShadow(
                    color: AppTheme.accentEmerald.withValues(alpha: 0.15),
                    blurRadius: 20,
                    offset: const Offset(0, 8),
                  ),
                ],
              ),
              child: Row(
                children: [
                  Container(
                    width: 60,
                    height: 60,
                    decoration: BoxDecoration(
                      color: Colors.white.withValues(alpha: 0.1),
                      borderRadius: BorderRadius.circular(16),
                    ),
                    alignment: Alignment.center,
                    child: Text(arrow, style: const TextStyle(fontSize: 32)),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Text(
                              'PREDICTED DIRECTION',
                              style: GoogleFonts.outfit(
                                fontSize: 11,
                                fontWeight: FontWeight.w700,
                                letterSpacing: 1.2,
                                color: AppTheme.accentCyan,
                              ),
                            ),
                            if (hasTrueLabel)
                              Container(
                                padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                                decoration: BoxDecoration(
                                  color: (isCorrect ? AppTheme.accentEmerald : AppTheme.accentAmber).withValues(alpha: 0.2),
                                  borderRadius: BorderRadius.circular(6),
                                  border: Border.all(
                                    color: (isCorrect ? AppTheme.accentEmerald : AppTheme.accentAmber).withValues(alpha: 0.5),
                                  ),
                                ),
                                child: Text(
                                  isCorrect ? 'VERIFIED MATCH' : 'DISGUISED SHOT',
                                  style: GoogleFonts.robotoMono(
                                    fontSize: 9,
                                    fontWeight: FontWeight.w700,
                                    color: isCorrect ? AppTheme.accentEmerald : AppTheme.accentAmber,
                                  ),
                                ),
                              ),
                          ],
                        ),
                        const SizedBox(height: 2),
                        Text(
                          result.direction.toUpperCase(),
                          style: GoogleFonts.outfit(
                            fontSize: 26,
                            fontWeight: FontWeight.w800,
                            color: Colors.white,
                          ),
                        ),
                        const SizedBox(height: 4),
                        Row(
                          children: [
                            Text(
                              'Confidence: ',
                              style: GoogleFonts.inter(fontSize: 13, color: AppTheme.textMuted),
                            ),
                            Text(
                              '${result.confidence.toStringAsFixed(1)}%',
                              style: GoogleFonts.robotoMono(
                                fontSize: 13,
                                fontWeight: FontWeight.w700,
                                color: AppTheme.accentEmerald,
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),

            // Video Replay & Visual Proof Card
            if (videoFile != null || (videoUrl != null && videoUrl!.isNotEmpty)) ...[
              VideoReplayCard(
                videoFile: videoFile,
                videoUrl: videoUrl,
                sampleName: sampleName,
                trueDirection: trueDirection,
                predictedDirection: result.direction,
              ),
              const SizedBox(height: 16),
            ],

            // Goal Net Visualizer
            GoalVisualizer(
              predictedDirection: result.direction,
              probabilities: result.probabilities,
            ),
            const SizedBox(height: 16),

            // Power Meter
            PowerMeter(
              power: result.power,
              category: result.powerCategory,
            ),
            const SizedBox(height: 16),

            // Strike Moment Pose Snapshot
            if (result.skeletonBytes != null) ...[
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: AppTheme.bgSurface,
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(color: AppTheme.borderSubtle),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text(
                          'CONTACT MOMENT SKELETAL POSE',
                          style: GoogleFonts.outfit(
                            fontSize: 12,
                            fontWeight: FontWeight.w700,
                            letterSpacing: 1.0,
                            color: AppTheme.textSubtle,
                          ),
                        ),
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                          decoration: BoxDecoration(
                            color: AppTheme.accentEmerald.withValues(alpha: 0.15),
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: Text(
                            'Frame #${result.strikeFrameIdx}',
                            style: GoogleFonts.robotoMono(
                              fontSize: 11,
                              fontWeight: FontWeight.w600,
                              color: AppTheme.accentEmerald,
                            ),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 12),
                    ClipRRect(
                      borderRadius: BorderRadius.circular(12),
                      child: AspectRatio(
                        aspectRatio: 16 / 9,
                        child: Image.memory(
                          result.skeletonBytes!,
                          fit: BoxFit.contain,
                        ),
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'Isolated run-up window up to foot-ball impact moment (MediaPipe 33 3D keypoints).',
                      style: GoogleFonts.inter(
                        fontSize: 11,
                        color: AppTheme.textSubtle,
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 16),
            ],

            // Biomechanics Grid
            if (result.biomechanics != null) ...[
              BiomechanicsGrid(biomechanics: result.biomechanics!),
              const SizedBox(height: 16),
            ],

            // Inference Diagnostics
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
              decoration: BoxDecoration(
                color: Colors.white.withValues(alpha: 0.03),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: AppTheme.borderSubtle),
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    'Multi-Scale Inception Inference',
                    style: GoogleFonts.inter(fontSize: 12, color: AppTheme.textMuted),
                  ),
                  Text(
                    '${result.inferenceTimeMs.toStringAsFixed(0)} ms',
                    style: GoogleFonts.robotoMono(
                      fontSize: 12,
                      fontWeight: FontWeight.w700,
                      color: AppTheme.accentCyan,
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 24),

            // Back / New Prediction Button
            ElevatedButton.icon(
              onPressed: () => Navigator.of(context).pop(),
              icon: const Icon(Icons.refresh_rounded),
              label: const Text('Analyze Another Kick'),
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 16),
              ),
            ),
            const SizedBox(height: 20),
          ],
        ),
      ),
    );
  }
}
