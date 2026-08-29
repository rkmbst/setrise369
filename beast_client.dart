// beast_client.dart
import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:sqflite/sqflite.dart';
import 'package:path/path.dart' as path;

class BeastClient {
  static final BeastClient _instance = BeastClient._internal();
  factory BeastClient() => _instance;
  BeastClient._internal();

  final List<Map<String, dynamic>> _buffer = [];
  Timer? _flushTimer;
  String _userId = 'anonymous';
  String _currentScreen = 'unknown';
  Database? _db;
  FlutterLocalNotificationsPlugin? _notificationsPlugin;

  // Change this to your Railway URL
  String serverUrl = 'https://your-app.up.railway.app';

  Future<void> init({required String userId, required BuildContext context}) async {
    _userId = userId;
    await _initDatabase();
    await _initNotifications();
    _flushTimer = Timer.periodic(Duration(seconds: 10), (_) => _flush());
    _track('session_start', {});
  }

  Future<void> _initDatabase() async {
    final dbPath = await getDatabasesPath();
    _db = await openDatabase(
      path.join(dbPath, 'beast_client.db'),
      version: 1,
      onCreate: (db, version) async {
        await db.execute('CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, event_json TEXT NOT NULL, sent INTEGER DEFAULT 0)');
      },
    );
  }

  Future<void> _initNotifications() async {
    _notificationsPlugin = FlutterLocalNotificationsPlugin();
    const androidSettings = AndroidInitializationSettings('@mipmap/ic_launcher');
    const iosSettings = IOSInitializationSettings();
    const initSettings = InitializationSettings(android: androidSettings, iOS: iosSettings);
    await _notificationsPlugin!.initialize(initSettings);
  }

  void _track(String type, Map<String, dynamic> data) {
    final event = {
      'event_type': type,
      'screen': _currentScreen,
      'data': data,
      'timestamp': DateTime.now().toIso8601String(),
    };
    _buffer.add(event);
    _db?.insert('events', {'event_json': jsonEncode(event), 'sent': 0});
    if (_buffer.length >= 100) _flush();
  }

  Future<void> _flush() async {
    if (_buffer.isEmpty) return;
    final events = List<Map<String, dynamic>>.from(_buffer);
    _buffer.clear();
    try {
      final response = await http.post(
        Uri.parse('$serverUrl/v5/events/batch'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'user_id': _userId, 'events': events}),
      ).timeout(Duration(seconds: 5));
      if (response.statusCode == 200) {
        await _db?.delete('events', where: 'sent = 0');
      } else {
        _buffer.addAll(events);
      }
    } catch (e) {
      _buffer.addAll(events);
    }
  }

  void setScreen(String screen) {
    if (_currentScreen != screen) {
      _track('screen_exit', {'duration_ms': DateTime.now().millisecondsSinceEpoch});
      _currentScreen = screen;
      _track('screen_view', {});
    }
  }

  void trackButton(String buttonName, {Map<String, dynamic>? extra}) {
    _track('button_click', {'button': buttonName, ...?extra});
  }

  void trackScroll(String direction, double offset) {
    _track('scroll', {'direction': direction, 'offset': offset});
  }

  void trackComment(String text) {
    _track('comment', {'text': text});
  }

  Future<List<Map<String, dynamic>>> getRecommendations({int topK = 20}) async {
    try {
      final response = await http.post(
        Uri.parse('$serverUrl/v5/recommend'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'user_id': _userId,
          'context': {
            'screen': _currentScreen,
            'hour': DateTime.now().hour,
            'device': Platform.isAndroid ? 'android' : 'ios',
          },
          'top_k': topK,
        }),
      ).timeout(Duration(seconds: 10));
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return (data['recommendations'] as List).cast<Map<String, dynamic>>();
      }
    } catch (e) {}
    return [];
  }

  Future<void> showLocalNotification(String title, String body) async {
    if (_notificationsPlugin == null) return;
    const androidDetails = AndroidNotificationDetails('beast_channel', 'Beast Notifications', importance: Importance.high, priority: Priority.high);
    const iosDetails = IOSNotificationDetails();
    const details = NotificationDetails(android: androidDetails, iOS: iosDetails);
    await _notificationsPlugin!.show(0, title, body, details);
  }

  void dispose() {
    _flushTimer?.cancel();
    _db?.close();
  }
}
