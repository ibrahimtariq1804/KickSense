import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../theme/app_theme.dart';

class GoalVisualizer extends StatelessWidget {
  final String predictedDirection; // "Left", "Center", "Right"
  final Map<String, double> probabilities;

  const GoalVisualizer({
    super.key,
    required this.predictedDirection,
    required this.probabilities,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: AppTheme.bgSurface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppTheme.borderSubtle),
      ),
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                'GOAL TARGET TRAJECTORY',
                style: GoogleFonts.outfit(
                  fontSize: 12,
                  fontWeight: FontWeight.w700,
                  letterSpacing: 1.0,
                  color: AppTheme.accentCyan,
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                decoration: BoxDecoration(
                  color: AppTheme.accentEmerald.withValues(alpha: 0.15),
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(color: AppTheme.accentEmerald.withValues(alpha: 0.4)),
                ),
                child: Text(
                  'Active: $predictedDirection',
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
          // Goal Post Frame
          Container(
            height: 130,
            decoration: BoxDecoration(
              color: Colors.black.withValues(alpha: 0.4),
              border: const Border(
                top: BorderSide(color: Colors.white, width: 4),
                left: BorderSide(color: Colors.white, width: 4),
                right: BorderSide(color: Colors.white, width: 4),
              ),
              borderRadius: const BorderRadius.only(
                topLeft: Radius.circular(4),
                topRight: Radius.circular(4),
              ),
            ),
            child: Row(
              children: [
                _buildZone('Left', probabilities['Left'] ?? 0.0),
                const VerticalDivider(color: Colors.white24, width: 1, thickness: 1),
                _buildZone('Center', probabilities['Center'] ?? 0.0),
                const VerticalDivider(color: Colors.white24, width: 1, thickness: 1),
                _buildZone('Right', probabilities['Right'] ?? 0.0),
              ],
            ),
          ),
          // Pitch Grass Line
          Container(
            height: 4,
            decoration: const BoxDecoration(
              gradient: LinearGradient(
                colors: [AppTheme.accentEmerald, Color(0xFF34D399), AppTheme.accentEmerald],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildZone(String zoneName, double prob) {
    final isSelected = predictedDirection.toLowerCase() == zoneName.toLowerCase();

    return Expanded(
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 300),
        decoration: BoxDecoration(
          color: isSelected
              ? AppTheme.accentEmerald.withValues(alpha: 0.28)
              : Colors.transparent,
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(
              zoneName.toUpperCase(),
              style: GoogleFonts.outfit(
                fontSize: 13,
                fontWeight: FontWeight.w700,
                letterSpacing: 0.8,
                color: isSelected ? Colors.white : AppTheme.textSubtle,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              '${prob.toStringAsFixed(0)}%',
              style: GoogleFonts.robotoMono(
                fontSize: isSelected ? 18 : 14,
                fontWeight: FontWeight.w700,
                color: isSelected ? AppTheme.accentEmerald : AppTheme.textMuted,
              ),
            ),
            if (isSelected) ...[
              const SizedBox(height: 6),
              const Text('⚽', style: TextStyle(fontSize: 18)),
            ],
          ],
        ),
      ),
    );
  }
}
