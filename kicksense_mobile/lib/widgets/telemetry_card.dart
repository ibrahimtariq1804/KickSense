import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../models/prediction_result.dart';
import '../theme/app_theme.dart';

class BiomechanicsGrid extends StatelessWidget {
  final BiomechanicsData biomechanics;

  const BiomechanicsGrid({
    super.key,
    required this.biomechanics,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
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
                'STRIKER BIOMECHANICS',
                style: GoogleFonts.outfit(
                  fontSize: 12,
                  fontWeight: FontWeight.w700,
                  letterSpacing: 1.0,
                  color: AppTheme.textSubtle,
                ),
              ),
              const Icon(Icons.accessibility_new_rounded, size: 16, color: AppTheme.accentCyan),
            ],
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: _buildTile(
                  label: 'Torso Lean',
                  value: '${biomechanics.torsoLeanDeg.toStringAsFixed(1)}°',
                  hint: biomechanics.torsoLeanLabel,
                  valueColor: AppTheme.accentCyan,
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: _buildTile(
                  label: 'Pelvis Angle',
                  value: '${biomechanics.pelvisAngleDeg.toStringAsFixed(1)}°',
                  hint: biomechanics.pelvisAngleLabel,
                  valueColor: AppTheme.accentEmerald,
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          Row(
            children: [
              Expanded(
                child: _buildTile(
                  label: 'Ankle Speed',
                  value: '${biomechanics.kickingAnklePeakSpeed.toStringAsFixed(0)} px/s',
                  hint: 'Swing Momentum',
                  valueColor: AppTheme.accentAmber,
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: _buildTile(
                  label: 'Plant Offset',
                  value: biomechanics.plantToBallDist.toStringAsFixed(2),
                  hint: 'Support Stance',
                  valueColor: AppTheme.accentPurple,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildTile({
    required String label,
    required String value,
    required String hint,
    required Color valueColor,
  }) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.04),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: AppTheme.borderSubtle),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label.toUpperCase(),
            style: GoogleFonts.outfit(
              fontSize: 10,
              fontWeight: FontWeight.w600,
              color: AppTheme.textSubtle,
              letterSpacing: 0.5,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            value,
            style: GoogleFonts.robotoMono(
              fontSize: 15,
              fontWeight: FontWeight.w700,
              color: valueColor,
            ),
          ),
          const SizedBox(height: 2),
          Text(
            hint,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: GoogleFonts.inter(
              fontSize: 11,
              color: AppTheme.textMuted,
            ),
          ),
        ],
      ),
    );
  }
}
