import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

/// Thrown when the app cannot get a Supabase user to save under (offline,
/// or anonymous sign-ins are switched off for the project).
class NotSignedInException implements Exception {
  final String message;
  const NotSignedInException([this.message = 'Could not sign in to save your rooms.']);

  @override
  String toString() => message;
}

/// Who the current user is. There is no login screen: each install signs in
/// anonymously once (Supabase Anonymous Auth) and keeps that identity on the
/// device, which is what lets saved rooms be private to a user and capped at a
/// fixed number per user. Reinstalling or clearing app data creates a new
/// anonymous user (the old rooms stay in the database but are no longer
/// reachable from the app).
class AuthService {
  AuthService({GoTrueClient? auth}) : _injected = auth;

  final GoTrueClient? _injected;
  Future<String>? _signingIn;

  GoTrueClient get _auth => _injected ?? Supabase.instance.client.auth;

  String? get currentUserId => _auth.currentUser?.id;

  /// The signed-in user's id, signing in anonymously first if there is no
  /// session yet. Concurrent callers share one sign-in (two at once would
  /// create two different users).
  Future<String> ensureSignedIn() {
    final existing = _auth.currentUser;
    if (existing != null) return Future.value(existing.id);

    return _signingIn ??= _signInAnonymously().whenComplete(() => _signingIn = null);
  }

  Future<String> _signInAnonymously() async {
    try {
      final response = await _auth.signInAnonymously();
      final user = response.user;
      if (user == null) throw const NotSignedInException();
      return user.id;
    } on AuthException catch (error) {
      throw NotSignedInException('Could not sign in to save your rooms: ${error.message}');
    }
  }
}

final authServiceProvider = Provider<AuthService>((ref) => AuthService());
