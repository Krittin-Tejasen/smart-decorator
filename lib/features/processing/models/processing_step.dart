
class ProcessingStep {

  final String title;
  final bool isCompleted;
  final bool isActive;

  ProcessingStep({
    required this.title,
    required this.isCompleted,
    required this.isActive,
  });

  ProcessingStep copyWith({
    String? title,
    bool? isCompleted,
    bool? isActive,
  }) {
    return ProcessingStep(
      title: title ?? this.title,
      isCompleted: isCompleted ?? this.isCompleted,
      isActive: isActive ?? this.isActive,
    );
  }
}