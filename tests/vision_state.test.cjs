const { test } = require('node:test');
const assert = require('node:assert/strict');
const { present, render } = require('../dashboard/vision_state.js');

function packet(changes = {}) {
  return {
    frame: { frame_id: 7 },
    vision: { frame_id: 7 },
    reasoning: {
      frame_id: 7, timestamp: '2026-10-23T12:00:00Z', state: 'TARGET_VERIFIED',
      target: { id: 'F7-B1', type: 'person', confidence: 0.94 },
      confidence: { score: 0.94, status: 'HIGH_CONFIDENCE' },
      hazard: { detected: false, type: null, confidence: null, detections: [] },
      location: { lat: 14.6, lon: 120.98 }, drone_coordinates: null, is_occluded: false,
      ...changes,
    },
  };
}

test('renders the five fields of a verified target', () => {
  const view = present(packet());
  assert.equal(view.visionTarget, 'PERSON');
  assert.equal(view.visionConfidence, '94.0%');
  assert.equal(view.visionHazard, 'None detected');
  assert.equal(view.visionLocation, '14.60000, 120.98000');
  assert.equal(view.visionState, 'TARGET VERIFIED');
  assert.match(view.visionSummaryBadge, /FRAME 7/);
});

test('retains hazard visibility during low confidence and occlusion', () => {
  const view = present(packet({state: 'LOW_CONFIDENCE', is_occluded: true,
    confidence: {score: 0.4, status: 'LOW_CONFIDENCE'},
    hazard: {detected: true, type: 'hazard', confidence: 0.9, detections: [{}]},
  }));
  assert.equal(view.visionState, 'LOW CONFIDENCE');
  assert.equal(view.visionHazard, 'HAZARD DETECTED');
  assert.equal(view.visionConfidence, '40.0%');
  assert.match(view.visionStateDetail, /human review/);
});

test('distinguishes an empty scene from absent vision', () => {
  const view = present(packet({state: 'NO_TARGET', target: null, location: null,
    confidence: {score: null, status: 'NO_DETECTION'}}));
  assert.equal(view.visionState, 'NO TARGET');
  assert.equal(view.visionTarget, 'None detected');
  assert.equal(view.visionConfidence, '—');
  assert.equal(present(null).visionState, 'AWAITING VISION');
  assert.equal(present(null).visionHazard, 'Unknown');
});

test('does not label drone GPS as a detection location', () => {
  const view = present(packet({location: null, drone_coordinates: {lat: 15.0, lon: 121.0}}));
  assert.equal(view.visionLocation, 'Location unavailable');
  assert.match(view.visionLocationDetail, /Drone GPS.*detection fix unavailable/);
});

test('discards mismatched reasoning and stale coordinates', () => {
  const input = packet();
  input.frame.frame_id = 8;
  assert.equal(present(input).visionTarget, 'Unavailable');
  assert.equal(present(input).visionHazard, 'Unknown');
  const stale = present(packet(), {stale: true});
  assert.equal(stale.visionState, 'STALE DATA');
  assert.equal(stale.visionLocation, 'Location unavailable');
  assert.equal(stale.visionTarget, 'Unavailable');
});

test('a zero confidence score is a measurement, not missing data', () => {
  const view = present(packet({confidence: {score: 0, status: 'LOW_CONFIDENCE'}}));
  assert.equal(view.visionConfidence, '0.0%');
});

test('renders through textContent and updates the state style', () => {
  const elements = new Map();
  const document = {getElementById(id) {
    if (!elements.has(id)) elements.set(id, {textContent: '', dataset: {}});
    return elements.get(id);
  }};
  render(packet(), {}, document);
  assert.equal(elements.get('visionState').textContent, 'TARGET VERIFIED');
  assert.equal(elements.get('visionSummary').dataset.state, 'TARGET_VERIFIED');
  render(null, {stale: true}, document);
  assert.equal(elements.get('visionLocation').textContent, 'Location unavailable');
  assert.equal(elements.get('visionSummary').dataset.state, 'STALE');
});
