/* Client for the stdlib JSON API in webapp/server.py. */

const JSON_HEADERS = { 'Content-Type': 'application/json' };

export class ApiError extends Error {
  constructor(message, status, details) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.details = details || [];
  }
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(path, options);
  } catch (error) {
    throw new ApiError('Cannot reach the local server. Is it still running?', 0, []);
  }

  const raw = await response.text();
  let payload = null;
  if (raw) {
    try {
      payload = JSON.parse(raw);
    } catch {
      payload = null;
    }
  }

  if (!response.ok) {
    const message = payload && payload.error ? payload.error : `Request failed (${response.status})`;
    const details = payload && Array.isArray(payload.details) ? payload.details : [];
    throw new ApiError(message, response.status, details);
  }
  return payload;
}

export const getOverview = () => request('/api/overview');
export const getSignals = () => request('/api/signals');
export const getPerformance = () => request('/api/performance');
export const getMethod = () => request('/api/method');
export const getDefaults = () => request('/api/defaults');

export const postPredict = (values) =>
  request('/api/predict', {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify({ values }),
  });
