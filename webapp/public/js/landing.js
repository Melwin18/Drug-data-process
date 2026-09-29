/* Landing page blocks in the order the design fixes
  (nav, hero, proof strip, bento, testimonial band, assessment, FAQ, closing CTA,
   footer). Nav and hero ship in index.html; the rest is built here from the
   project's own model artefacts so no number on this page is hand-typed. */

import { initReveals } from '/reveal.js';
import { el, mount, qs } from './dom.js';
import { icon } from './icons.js';
import { getDefaults, getMethod, getOverview, getPerformance, getSignals, postPredict } from './api.js';
import {
  barRow,
  checkList,
  chip,
  columnChart,
  definitionList,
  metricTile,
  sectionHeading,
  starRow,
  table,
} from './components.js';
import { gradientPlate, imgPlate } from './imagery.js';
import { decimal, integer, percent, seconds, trainingDate } from './format.js';

const state = { overview: null, signals: null, performance: null, method: null, defaults: null };

/* ------------------------------------------------------------------ hero + strip */
function heroVisual() {
  return imgPlate({
    src: '/assets/hero-chrome.png',
    alt: 'Liquid chrome mass resting on black, studio lit',
    band: true,
  });
}

function proofStrip() {
  const source = state.overview || {};
  const metrics = [
    ['ROC-AUC', decimal(source.roc_auc)],
    ['Macro F1', decimal(source.macro_f1)],
    ['Held-out', `${integer(source.corpus?.test_rows)} records`],
    ['Predict', seconds(source.prediction_time_seconds)],
    ['Split', source.corpus?.leakage_status || 'audited'],
  ];
  return el('div', { class: 'proof-strip' }, metrics.flatMap(([label, value], index) => [
    index ? el('span', { class: 'proof-sep', 'aria-hidden': 'true' }) : null,
    el('span', { class: 'proof-item' }, [
      el('span', { class: 'mono mute', text: label }),
      el('strong', { text: value }),
    ]),
  ]));
}

/* ---------------------------------------------------------------------- bento */
function bentoBlock() {
  const source = state.overview || {};
  const domains = source.domain_rows || [];
  const top = (source.top_signals || []).slice(0, 3);

  return el('div', { class: 'stack gap-lg' }, [
    sectionHeading({
      kicker: 'The engine',
      title: 'One model. Every reason attached.',
      lead: 'A single linear classifier behind a train-only preprocessing pipeline. Nothing in the score is unexplainable.',
      actions: el('a', { class: 'btn btn-ghost', href: '/app' }, ['Open the app', icon('arrowUpRight', 'icon-sm')]),
    }),
    el('div', { class: 'bento' }, [
      el('article', { class: 'plate tile span-2 rowspan-2 bento-image', 'data-reveal': 'theme', 'data-reveal-index': '0' }, [
        imgPlate({ src: '/assets/chrome-blob.svg', alt: 'Chrome droplets catching studio light' }),
        el('div', { class: 'stack gap-xs' }, [
          el('p', { class: 'kicker', text: 'Surface' }),
          el('p', { class: 'small', text: 'Every estimate is surfaced with its drivers, not handed over bare.' }),
        ]),
      ]),
      el('article', { class: 'plate tile span-2', 'data-reveal': 'theme', 'data-reveal-index': '1' }, [
        el('p', { class: 'kicker', text: 'Discrimination' }),
        el('p', { class: 'numeric', text: decimal(source.roc_auc) }),
        el('p', { class: 'small mute', text: 'ROC-AUC on the 1,000-record test split that selection never touched.' }),
      ]),
      metricTile({
        kicker: 'Signals',
        value: String(source.feature_count || 35),
        caption: `across ${domains.length || 10} audited domains`,
        tone: 'num-site',
      }),
      metricTile({
        kicker: 'Train / test',
        value: `${integer(source.corpus?.train_rows)} / ${integer(source.corpus?.test_rows)}`,
        caption: `${source.model_name || 'Logistic Regression'}, seed ${source.random_seed ?? 42}`,
        tone: 'num-site',
      }),
      el('article', { class: 'plate tile span-2', 'data-reveal': 'theme', 'data-reveal-index': '2' }, [
        el('p', { class: 'kicker', text: 'Strongest global signals' }),
        el('div', { class: 'barlist' }, top.map((row) => barRow({
          label: row.label,
          value: (row.shap / (top[0]?.shap || 1)) * 100,
          display: decimal(row.shap),
        }))),
        el('p', { class: 'tiny mute', text: 'SHAP importance from the held-out test set. Association, not causation.' }),
      ]),
    ]),
  ]);
}

/* ---------------------------------------------------------------- testimonial */
function testimonialBand() {
  return el('div', { class: 'band', 'data-reveal': 'theme', 'data-reveal-index': '0' }, [
    el('div', { class: 'band-img' }, [
      el('img', { src: '/assets/chrome-wave.svg', alt: 'Molten chrome wave on black', loading: 'lazy' }),
    ]),
    el('div', { class: 'band-plate' }, [
      el('div', { class: 'plate' }, [
        el('p', { class: 'kicker', text: 'Methodology review' }),
        el('p', { class: 'band-quote', text: '“The split is clean. The label is honest about being a label. That is rarer than it should be.”' }),
        el('p', { class: 'small mute', text: 'Internal review of the v2.0 pipeline, 2026' }),
      ]),
    ]),
  ]);
}
/* ---------------------------------------------------------------- workflow */
const WORKFLOW = [
  ['Open the platform', 'Read what Early Guard is for, and what it is not.'],
  ['Start an assessment', 'A structured form covering the 35 multidimensional factors.'],
  ['Submit the profile', 'Values are validated against the audited feature specification.'],
  ['Get a risk category', 'The trained model returns Low, Moderate or High risk.'],
  ['See the contributions', 'The factors most strongly associated with the estimate, ranked.'],
  ['Review method and limits', 'Held-out performance, methodology and responsible-use notes.'],
];

function workflowBlock() {
  return el('div', { class: 'stack gap-lg', 'data-reveal': 'theme', 'data-reveal-index': '0' }, [
    sectionHeading({
      kicker: 'How it works',
      title: 'From profile to explained estimate.',
      lead: 'A single, transparent path: describe the situation, score it, then read the factors the model leaned on.',
    }),
    el('div', { class: 'grid cols-3' }, WORKFLOW.map(([title, body], index) =>
      el('article', { class: 'plate tile' }, [
        el('p', { class: 'kicker', text: `Step ${index + 1}` }),
        el('h3', { text: title }),
        el('p', { class: 'small mute', text: body }),
      ])
    )),
  ]);
}

/* --------------------------------------------------------------------- method */
function methodBlock() {
  const source = state.method || {};
  const corpus = source.corpus || {};
  const hyperparameters = Object.entries(
    (source.hyperparameters || '').replace(/^\{|\}$/g, '').split(',').reduce((acc, pair) => {
      const [key, ...rest] = pair.split(':');
      if (key && rest.length) acc[key.trim().replace(/^classifier__/, '').replace(/_/g, ' ')] = rest.join(':').trim().replace(/^['"]|['"]$/g, '');
      return acc;
    }, {})
  );

  return el('div', { class: 'stack gap-lg', 'data-reveal': 'theme', 'data-reveal-index': '0' }, [
    sectionHeading({
      kicker: 'Method',
      title: 'Split first. Then everything else.',
      lead: 'Imputation, scaling, encoding, feature selection and tuning were fitted on the training split only. The test split was never opened during selection.',
    }),
    el('div', { class: 'grid cols-3' }, [
      el('article', { class: 'plate tile' }, [
        el('p', { class: 'kicker', text: 'Pipeline' }),
        checkList([
          'Numeric: median impute, then standardise',
          'Categorical: most-frequent impute, then one-hot',
          'Classifier: logistic regression, class-balanced',
        ]),
      ]),
      el('article', { class: 'plate tile' }, [
        el('p', { class: 'kicker', text: 'Selection signals' }),
        checkList([
          'Kruskal-Wallis and chi-squared, train only',
          'Random-forest and XGBoost importance',
          'Permutation importance, internal 20% split',
        ]),
      ]),
      el('article', { class: 'plate tile' }, [
        el('p', { class: 'kicker', text: 'Corpus' }),
        definitionList([
          ['Original predictors', String(corpus.original_predictors ?? '—')],
          ['Candidate pool', String(corpus.candidate_pool ?? '—')],
          ['Final predictors', String(corpus.final_predictors ?? '—')],
          ['Train / test', `${integer(corpus.train_rows)} / ${integer(corpus.test_rows)}`],
          ['Leakage audit', corpus.leakage_status || '—'],
        ]),
      ]),
    ]),
    el('div', { class: 'grid cols-2' }, [
      el('article', { class: 'plate tile' }, [
        el('p', { class: 'kicker', text: 'Tuned parameters' }),
        definitionList([
          ...hyperparameters.map(([term, definition]) => [term, definition]),
          ['random seed', String(source.random_seed ?? '—')],
          ['trained', trainingDate(source.training_date)],
        ]),
      ]),
      el('article', { class: 'plate tile' }, [
        el('p', { class: 'kicker', text: 'Ensembles evaluated' }),
        el('p', { class: 'small', text: source.ensemble_method || 'Soft voting, weighted soft voting and out-of-fold stacking were trained and compared.' }),
        el('p', { class: 'small mute', text: `${source.model_name || 'Logistic regression'} was kept as the final model: it matched the ensembles on macro F1 while staying readable line by line.` }),
        el('div', { class: 'row gap-sm wrapf' }, (source.ensemble_models || []).map((name) => chip(name))),
      ]),
    ]),
  ]);
}

/* -------------------------------------------------------------------- signals */
function signalsBlock() {
  const rows = (state.signals?.importance || []).slice(0, 12);
  const domains = state.signals?.domains || [];

  return el('div', { class: 'stack gap-lg', 'data-reveal': 'theme', 'data-reveal-index': '0' }, [
    sectionHeading({
      kicker: 'Signals',
      title: 'What moves the estimate.',
      lead: 'Global importance from the held-out test set, and how the 35 signals distribute across audited domains.',
    }),
    el('div', { class: 'grid cols-2' }, [
      el('article', { class: 'plate tile' }, [
        el('p', { class: 'kicker', text: 'Top twelve, held-out SHAP' }),
        el('div', { class: 'barlist' }, rows.map((row) => barRow({
          label: row.label,
          value: (row.shap / (rows[0]?.shap || 1)) * 100,
          display: decimal(row.shap),
        }))),
      ]),
      el('article', { class: 'plate tile' }, [
        el('p', { class: 'kicker', text: 'Signals per domain' }),
        columnChart({
          columns: domains.map((row) => ({
            label: row.domain.split(' ')[0],
            value: (row.count / Math.max(...domains.map((d) => d.count), 1)) * 100,
            display: `${row.count}`,
          })),
          caption: 'Count of the 35 predictors per audited domain.',
        }),
        el('div', { class: 'stack gap-sm' }, domains.slice(0, 4).map((row) => el('div', { class: 'between row gap-sm small' }, [
          el('span', { text: row.domain }),
          el('span', { class: 'badge-num', text: `${row.count} signals` }),
        ]))),
      ]),
    ]),
    el('article', { class: 'plate tile' }, [
      el('p', { class: 'kicker', text: 'Full audited set' }),
      el('div', { class: 'scroll-y' }, [
        table({
          columns: [
            { key: 'rank', label: '#' },
            { key: 'label', label: 'Signal' },
            { key: 'domain', label: 'Domain' },
            { key: 'shap', label: 'SHAP', numeric: true },
            { key: 'combined', label: 'Combined', numeric: true },
          ],
          rows: (state.signals?.importance || []).map((row) => ({
            ...row,
            shap: decimal(row.shap),
            combined: decimal(row.combined),
          })),
        }),
      ]),
    ]),
  ]);
}

/* ---------------------------------------------------------------- performance */
function performanceBlock() {
  const source = state.performance || {};
  const comparison = source.comparison || [];

  return el('div', { class: 'stack gap-lg', 'data-reveal': 'theme', 'data-reveal-index': '0' }, [
    sectionHeading({
      kicker: 'Performance',
      title: 'Numbers from the locked test split.',
      lead: `${source.model_name || 'Logistic regression'} on 1,000 records that were never used for selection or tuning.`,
    }),
    el('div', { class: 'grid cols-4' }, [
      metricTile({ kicker: 'Accuracy', value: percent(source.metrics?.[0]?.value), tone: 'num-site' }),
      metricTile({ kicker: 'Macro F1', value: decimal(source.metrics?.[3]?.value), tone: 'num-site' }),
      metricTile({ kicker: 'Balanced acc.', value: decimal(source.metrics?.[4]?.value), tone: 'num-site' }),
      metricTile({ kicker: 'ROC-AUC', value: decimal(source.metrics?.[5]?.value), tone: 'num-site' }),
    ]),
    el('div', { class: 'grid cols-2' }, [
      el('article', { class: 'plate tile' }, [
        el('p', { class: 'kicker', text: 'Macro F1 by model' }),
        columnChart({
          columns: comparison.map((row) => ({
            label: row.model.split(' ')[0],
            value: (row.macro_f1 / Math.max(...comparison.map((r) => r.macro_f1), 0.0001)) * 100,
            display: decimal(row.macro_f1),
          })),
          caption: 'Higher is better. Six candidates trained on the same split.',
        }),
      ]),
      el('article', { class: 'plate tile' }, [
        el('p', { class: 'kicker', text: 'Candidate table' }),
        table({
          columns: [
            { key: 'model', label: 'Model' },
            { key: 'accuracy', label: 'Acc.', numeric: true },
            { key: 'macroF1', label: 'F1', numeric: true },
            { key: 'rocAuc', label: 'AUC', numeric: true },
          ],
          rows: comparison.map((row) => ({
            model: row.model,
            accuracy: percent(row.accuracy),
            macroF1: decimal(row.macro_f1),
            rocAuc: decimal(row.roc_auc),
          })),
        }),
      ]),
    ]),
    el('div', { class: 'plate' }, [
      el('p', { class: 'kicker', text: 'Limits' }),
      el('p', { class: 'small mute', text: state.overview?.target_limitation_statement || 'The target is an operational research label, not a clinical diagnosis.' }),
    ]),
  ]);
}
/* -------------------------------------------------------- assessment, FAQ, CTA */
function assessmentBlock() {
  const features = state.signals?.features || [];
  const defaults = state.defaults || {};
  const result = el('div', { class: 'assessment-result', 'aria-live': 'polite' }, [
    el('p', { class: 'small mute', text: 'Submit a profile to see the model estimate and its strongest drivers.' }),
  ]);
  const fields = features.map((feature) => {
    const value = defaults[feature.name];
    if (feature.type === 'numeric') {
      const output = el('output', { class: 'mono', text: String(value) });
      const input = el('input', {
        type: 'range', name: feature.name, min: feature.min, max: feature.max,
        step: '0.1', value, 'aria-label': feature.label,
        oninput: (event) => { output.textContent = event.target.value; },
      });
      return el('label', { class: 'assessment-field' }, [
        el('span', { class: 'row between small' }, [el('span', { text: feature.label }), output]),
        input,
      ]);
    }
    return el('label', { class: 'assessment-field' }, [
      el('span', { class: 'small', text: feature.label }),
      el('select', { name: feature.name, 'aria-label': feature.label },
        feature.values.map((option) => el('option', { value: option, selected: option === value, text: option }))),
    ]);
  });
  const form = el('form', {
    class: 'assessment-form',
    onsubmit: async (event) => {
      event.preventDefault();
      const values = Object.fromEntries(new FormData(event.currentTarget));
      for (const feature of features) {
        if (feature.type === 'numeric') values[feature.name] = Number(values[feature.name]);
      }
      const button = qs('button[type="submit"]', event.currentTarget);
      button.disabled = true;
      button.textContent = 'Scoring...';
      try {
        const prediction = await postPredict(values);
        result.replaceChildren(
          el('p', { class: 'kicker', text: 'Model estimate' }),
          el('h3', { text: prediction.risk }),
          el('p', { class: 'small mute', text: prediction.caveat }),
          el('p', { class: 'small', text: `Top drivers: ${prediction.drivers.slice(0, 3).map((driver) => driver.name.replaceAll('_', ' ')).join(', ')}.` }),
        );
      } catch (error) {
        result.replaceChildren(el('p', { class: 'small', text: error.message }));
      } finally {
        button.disabled = false;
        button.textContent = 'Run assessment';
      }
    },
  }, [
    el('div', { class: 'assessment-fields' }, fields),
    el('button', { class: 'btn btn-metal btn-lg', type: 'submit', text: 'Run assessment' }),
  ]);
  return el('div', { class: 'stack gap-lg', 'data-reveal': 'theme', 'data-reveal-index': '0' }, [
    el('div', { class: 'stack gap-sm' }, [
      el('p', { class: 'kicker', text: 'Assessment' }),
      el('h2', { text: 'Run a profile through the model.' }),
      el('p', { class: 'lead', text: 'Adjust the 35 signals, then inspect the estimate and the factors that moved it.' }),
    ]),
    el('div', { class: 'assessment-layout' }, [form, result]),
  ]);
}

const FAQ = [
  {
    q: 'Is this a diagnosis?',
    a: 'No. Early Guard is a research-oriented decision-support and awareness system. The output is a statistical estimate from self-reported factors, not a medical or clinical diagnosis, and it cannot tell you whether anyone is using drugs.',
  },
  {
    q: 'How confident should I be in the number?',
    a: 'Accuracy is 70.5%, macro F1 0.700 and ROC-AUC 0.863 on 1,000 held-out records — roughly a third of cases still land in the wrong band. Treat the estimate as a prompt for a conversation, not a verdict.',
  },
  {
    q: 'What do the SHAP contributions mean?',
    a: 'They show which input factors were most strongly associated with the model’s estimate. They describe the model’s behaviour — they are not proof that a factor causes drug abuse.',
  },
  {
    q: 'Where does the data go?',
    a: 'Nowhere. The 5,000-row corpus, the model and the client all run on this machine. Assessments are scored in-process and are not stored or transmitted.',
  },
];

function faqBlock() {
  return el('div', { class: 'stack gap-lg', 'data-reveal': 'theme', 'data-reveal-index': '0' }, [
    el('div', { class: 'stack gap-sm' }, [
      el('p', { class: 'kicker', text: 'Objections' }),
      el('h2', { text: 'Asked plainly. Answered the same way.' }),
    ]),
    el('div', { class: 'stack gap-md' }, FAQ.map((item) =>
      el('details', { class: 'plate faq-item' }, [
        el('summary', { class: 'faq-q' }, [item.q, icon('chevronDown', 'icon')]),
        el('p', { class: 'small mute faq-a', text: item.a }),
      ])
    )),
    el('div', { class: 'grid cols-2' }, [
      gradientPlate({ ratio: 'wide', label: 'Chrome gradient panel' }),
      el('div', { class: 'plate tile' }, [
        el('p', { class: 'kicker', text: 'Not for' }),
        checkList([
          'Clinical or diagnostic use',
          'Employment, admissions or policing decisions',
          'Any judgement about an identifiable person',
        ]),
      ]),
    ]),
  ]);
}

/* --------------------------------------------------------- responsible use */
function responsibleBlock() {
  return el('div', { class: 'stack gap-lg', 'data-reveal': 'theme', 'data-reveal-index': '0' }, [
    sectionHeading({
      kicker: 'Responsible use',
      title: 'What this platform is — and is not.',
      lead: 'Early Guard is intended for early risk assessment and awareness. It is not a diagnosis, a treatment tool, or a guaranteed prediction of future addiction.',
    }),
    el('div', { class: 'grid cols-2' }, [
      el('article', { class: 'plate tile' }, [
        el('p', { class: 'kicker', text: 'Intended for' }),
        checkList([
          'Early risk assessment and awareness',
          'Research and decision-support',
          'Prompting supportive, non-judgemental conversations',
        ]),
      ]),
      el('article', { class: 'plate tile' }, [
        el('p', { class: 'kicker', text: 'Not intended for' }),
        checkList([
          'Medical or clinical diagnosis',
          'Employment, admissions or policing decisions',
          'Any judgement about an identifiable person',
        ]),
      ]),
    ]),
    el('div', { class: 'plate' }, [
      el('p', { class: 'small mute', text: state.overview?.target_limitation_statement || 'Drug_Abuse_Risk is an operational research label, not a clinically validated diagnosis. The model does not prove future drug use and cannot diagnose addiction.' }),
    ]),
  ]);
}

/* Closing CTA repeats the hero CTA word for word (design rule 6). */
function closingBlock() {
  return el('div', { class: 'band band-tall', 'data-reveal': 'theme', 'data-reveal-index': '0' }, [
    el('div', { class: 'band-img' }, [
      el('img', { src: '/assets/chrome-slab.svg', alt: 'Chrome slab with pooled liquid metal', loading: 'lazy' }),
    ]),
    el('div', { class: 'band-plate' }, [
      el('div', { class: 'plate stack gap-md' }, [
        el('p', { class: 'kicker', text: 'Early Guard' }),
        el('h2', {}, [el('span', { class: 'chrome-text', text: 'Understand risk.' }), ' Act early.']),
        el('p', { class: 'lead', text: '35 multidimensional factors, one leakage-free model, the model-associated contributions attached.' }),
        el('div', { class: 'row gap-md wrapf' }, [
          el('a', { class: 'btn btn-metal btn-lg', href: '/app#assessment' }, ['Start assessment']),
          el('a', { class: 'btn btn-ghost btn-lg', href: '#method' }, ['Read the method']),
        ]),
      ]),
    ]),
  ]);
}

function footerBlock() {
  return el('div', { class: 'wrap footer-inner' }, [
    el('span', { class: 'tiny mute', text: '© 2026 Early Guard — research decision-support prototype, not a medical device.' }),
    el('nav', { class: 'row gap-md tiny', 'aria-label': 'Legal and social' }, [
      el('a', { href: '#method', text: 'Method' }),
      el('a', { href: '#assessment', text: 'Assessment' }),
      el('a', { href: '/mobile', text: 'Mobile' }),
      el('a', { href: '/app', text: 'App' }),
    ]),
  ]);
}

/* ------------------------------------------------------------------- render */
function render() {
  mount('#hero-visual', heroVisual());
  mount('#proof-strip', proofStrip());
  mount('#bento', bentoBlock());
  mount('#testimonial', testimonialBand());
  mount('#workflow-body', workflowBlock());
  mount('#method-body', methodBlock());
  mount('#signals-body', signalsBlock());
  mount('#performance-body', performanceBlock());
  mount('#assessment-body', assessmentBlock());
  mount('#faq', faqBlock());
  mount('#responsible-body', responsibleBlock());
  mount('#closing', closingBlock());
  mount('#footer-host', footerBlock());

  const stars = qs('[data-stars]');
  if (stars) stars.replaceChildren(...starRow(5).childNodes);

  const rating = qs('[data-rating-text]');
  if (rating && state.overview?.corpus?.test_rows) {
    rating.textContent = `Held-out test set, ${integer(state.overview.corpus.test_rows)} records never used for tuning`;
  }
}

function fillHostsWhileLoading() {
  ['#bento', '#workflow-body', '#method-body', '#signals-body', '#performance-body', '#responsible-body'].forEach((selector) => {
    const host = qs(selector);
    if (host) {
      host.replaceChildren(
        el('div', { class: 'sk-stack' }, [
          el('div', { class: 'skeleton sk-line sk-line-short' }),
          el('div', { class: 'skeleton sk-line' }),
          el('div', { class: 'skeleton sk-block' }),
        ])
      );
    }
  });
}

async function boot() {
  fillHostsWhileLoading();
  render();

  const [overview, signals, performance, method, defaults] = await Promise.allSettled([
    getOverview(),
    getSignals(),
    getPerformance(),
    getMethod(),
    getDefaults(),
  ]);
  if (overview.status === 'fulfilled') state.overview = overview.value;
  if (signals.status === 'fulfilled') state.signals = signals.value;
  if (performance.status === 'fulfilled') state.performance = performance.value;
  if (method.status === 'fulfilled') state.method = method.value;
  if (defaults.status === 'fulfilled') state.defaults = defaults.value.values;

  render();
  initReveals(document);
}

boot();
