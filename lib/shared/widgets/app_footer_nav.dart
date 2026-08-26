import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

import '../../core/constants/app_colors.dart';

enum FooterTab { home, history, none }

/// Shared bottom footer (Home / History) used on the Home, Results and
/// History screens, so all three look and behave the same way.
class AppFooterNav extends StatelessWidget {
  final FooterTab current;

  const AppFooterNav({super.key, this.current = FooterTab.none});

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: AppColors.surface,
        border: Border(
          top: BorderSide(color: AppColors.ink.withValues(alpha: 0.06)),
        ),
      ),
      child: SafeArea(
        top: false,
        child: Padding(
          padding: const EdgeInsets.fromLTRB(12, 8, 12, 8),
          child: Row(
            children: [
              Expanded(
                child: _FooterItem(
                  icon: Icons.home_rounded,
                  label: 'Home',
                  active: current == FooterTab.home,
                  onTap: () {
                    if (current == FooterTab.home) return;
                    context.go('/home');
                  },
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: _FooterItem(
                  icon: Icons.history_rounded,
                  label: 'History',
                  active: current == FooterTab.history,
                  onTap: () {
                    if (current == FooterTab.history) return;
                    context.push('/history');
                  },
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _FooterItem extends StatelessWidget {
  final IconData icon;
  final String label;
  final bool active;
  final VoidCallback onTap;

  const _FooterItem({
    required this.icon,
    required this.label,
    required this.active,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(14),
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 180),
          padding: const EdgeInsets.symmetric(vertical: 8),
          decoration: BoxDecoration(
            color: active ? AppColors.sageTint : Colors.transparent,
            borderRadius: BorderRadius.circular(14),
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(
                icon,
                size: 22,
                color: active ? AppColors.sageDeep : AppColors.muted,
              ),
              const SizedBox(height: 2),
              Text(
                label,
                style: TextStyle(
                  fontSize: 11,
                  fontWeight: FontWeight.w600,
                  color: active ? AppColors.sageDeep : AppColors.muted,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
