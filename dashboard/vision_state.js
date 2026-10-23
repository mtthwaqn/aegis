/* Presentation only: all spatial decisions come from Member 3 on the server. */
(function (root) {
  'use strict';

  const percent = value => value === null ? '—' : (value * 100).toFixed(1) + '%';
  const coordinates = value => value.lat.toFixed(5) + ', ' + value.lon.toFixed(5);
  const words = value => value.replaceAll('_', ' ');

  function present(packet, { stale = false } = {}) {
    const view = {
      visionSummaryBadge: stale ? 'STALE / DISCONNECTED' : 'WAITING FOR MATCHING VISION',
      visionTarget: 'Unavailable',
      visionTargetDetail: 'Awaiting synchronized detections',
      visionConfidence: '—',
      visionConfidenceDetail: 'No current confidence assessment',
      visionHazard: 'Unknown',
      visionHazardDetail: 'Awaiting synchronized detections',
      visionLocation: 'Location unavailable',
      visionLocationDetail: 'No current detection GPS fix',
      visionState: stale ? 'STALE DATA' : 'AWAITING VISION',
      visionStateDetail: stale ? 'Previous detections cleared' : 'Awaiting Member 3 assessment',
      state: stale ? 'STALE' : 'WAITING',
    };
    const r = packet?.reasoning;
    if (stale || !r || !packet.frame || !packet.vision ||
        r.frame_id !== packet.frame.frame_id || r.frame_id !== packet.vision.frame_id) {
      return view;
    }

    view.visionSummaryBadge = 'FRAME ' + r.frame_id + ' / ' +
      new Date(r.timestamp).toLocaleTimeString();
    view.visionTarget = r.target ? r.target.type.toUpperCase() : 'None detected';
    view.visionTargetDetail = r.target
      ? r.target.id + ' / ' + percent(r.target.confidence) + ' target confidence'
      : 'No survivor in this frame';
    view.visionConfidence = percent(r.confidence.score);
    view.visionConfidenceDetail = words(r.confidence.status) +
      (r.confidence.score === null ? '' : ' / lowest detection score');
    view.visionHazard = r.hazard.detected ? r.hazard.type.toUpperCase() + ' DETECTED' : 'None detected';
    view.visionHazardDetail = r.hazard.detected
      ? r.hazard.detections.length + ' hazard/debris regions / ' + percent(r.hazard.confidence)
      : 'No hazard or debris in this frame';
    if (r.location) {
      view.visionLocation = coordinates(r.location);
      view.visionLocationDetail = 'Selected detection GPS';
    } else {
      view.visionLocationDetail = r.drone_coordinates
        ? 'Drone GPS ' + coordinates(r.drone_coordinates) + ' / detection fix unavailable'
        : 'Georeferencing metadata required';
    }
    view.visionState = words(r.state);
    view.visionStateDetail = r.is_occluded ? 'Occluded / human review required' : 'Member 3 spatial assessment';
    view.state = r.state;
    return view;
  }

  function render(packet, options, document) {
    const view = present(packet, options);
    for (const [id, value] of Object.entries(view)) {
      if (id !== 'state') document.getElementById(id).textContent = value;
    }
    document.getElementById('visionSummary').dataset.state = view.state;
  }

  const api = { present, render };
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.AegisVision = api;
})(globalThis);
