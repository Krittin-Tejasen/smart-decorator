import 'dart:async';

import 'package:flutter_test/flutter_test.dart';
import 'package:smart_decorator/core/services/auth_service.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

User userWithId(String id) => User(
      id: id,
      appMetadata: const {},
      userMetadata: const {},
      aud: 'authenticated',
      createdAt: '2026-09-27T00:00:00Z',
    );

/// Just enough of Supabase's auth client for AuthService.
class FakeAuth implements GoTrueClient {
  FakeAuth({this.user});

  User? user;
  int signInCalls = 0;
  Completer<AuthResponse>? gate;
  Object? failWith;
  bool returnNoUser = false;

  @override
  User? get currentUser => user;

  @override
  Future<AuthResponse> signInAnonymously({Map<String, dynamic>? data, String? captchaToken}) async {
    signInCalls++;
    if (gate != null) return gate!.future;
    if (failWith != null) throw failWith!;
    if (returnNoUser) return AuthResponse();
    user = userWithId('anon-$signInCalls');
    return AuthResponse(user: user);
  }

  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

void main() {
  test('an existing session is reused without signing in again', () async {
    final auth = FakeAuth(user: userWithId('existing'));

    expect(await AuthService(auth: auth).ensureSignedIn(), 'existing');
    expect(auth.signInCalls, 0);
  });

  test('with no session it signs in anonymously and returns the new id', () async {
    final auth = FakeAuth();
    final service = AuthService(auth: auth);

    expect(await service.ensureSignedIn(), 'anon-1');
    expect(service.currentUserId, 'anon-1');
    expect(auth.signInCalls, 1);
  });

  test('callers at the same time share one sign-in (two would make two users)', () async {
    final auth = FakeAuth()..gate = Completer<AuthResponse>();
    final service = AuthService(auth: auth);

    final first = service.ensureSignedIn();
    final second = service.ensureSignedIn();
    final third = service.ensureSignedIn();
    auth.user = userWithId('the-one');
    auth.gate!.complete(AuthResponse(user: auth.user));

    expect(await Future.wait([first, second, third]), ['the-one', 'the-one', 'the-one']);
    expect(auth.signInCalls, 1);
  });

  test('a rejected sign-in (e.g. anonymous sign-ins switched off) is a clear error', () async {
    final auth = FakeAuth()..failWith = const AuthException('Anonymous sign-ins are disabled');

    await expectLater(
      AuthService(auth: auth).ensureSignedIn(),
      throwsA(isA<NotSignedInException>().having(
        (e) => e.message,
        'message',
        contains('Anonymous sign-ins are disabled'),
      )),
    );
  });

  test('after a failure the next call tries again instead of caching the failure', () async {
    final auth = FakeAuth()..failWith = const AuthException('offline');
    final service = AuthService(auth: auth);
    await expectLater(service.ensureSignedIn(), throwsA(isA<NotSignedInException>()));

    auth.failWith = null;

    expect(await service.ensureSignedIn(), 'anon-2');
    expect(auth.signInCalls, 2);
  });

  test('a response without a user is a failure, not a null id', () async {
    final auth = FakeAuth()..returnNoUser = true;

    await expectLater(AuthService(auth: auth).ensureSignedIn(), throwsA(isA<NotSignedInException>()));
  });
}
