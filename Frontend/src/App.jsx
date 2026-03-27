import { useMemo, useState } from "react";
import { getApiBaseUrl, getErrorMessage } from "./api";

const SCORE_WEIGHTS = [
  { key: "skills_score", label: "Skills", max: 40 },
  { key: "experience_score", label: "Experience", max: 30 },
  { key: "education_score", label: "Education", max: 15 },
  { key: "keyword_score", label: "Keywords", max: 10 },
  { key: "completeness_score", label: "Completeness", max: 5 },
];

const API_BASE_URL = getApiBaseUrl();

function formatScore(value, max) {
  const safeValue = typeof value === "number" ? value : 0;
  return `${safeValue}/${max}`;
}

function ResultPill({ text }) {
  return <span className="pill">{text}</span>;
}

function ScoreRow({ label, value, max }) {
  const percent = Math.min(Math.round(((value || 0) / max) * 100), 100);
  return (
    <div className="score-row">
      <div className="score-row-head">
        <span>{label}</span>
        <span>{formatScore(value, max)}</span>
      </div>
      <div className="score-track">
        <div className="score-bar" style={{ width: `${percent}%` }} />
      </div>
    </div>
  );
}

export default function App() {
  const [jobDescription, setJobDescription] = useState("");
  const [topK, setTopK] = useState(5);
  const [resumeFiles, setResumeFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [errorText, setErrorText] = useState("");
  const [responseData, setResponseData] = useState(null);
  const [selectedIndex, setSelectedIndex] = useState(0);

  const selectedResult = useMemo(() => {
    if (
      !responseData ||
      !responseData.results ||
      !responseData.results.length
    ) {
      return null;
    }
    return responseData.results[
      Math.min(selectedIndex, responseData.results.length - 1)
    ];
  }, [responseData, selectedIndex]);

  async function handleSubmit(event) {
    event.preventDefault();
    setErrorText("");

    if (!jobDescription.trim()) {
      setErrorText("Job description is required.");
      return;
    }

    if (!resumeFiles.length) {
      setErrorText("Upload at least one resume.");
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append("job_description", jobDescription.trim());
    formData.append("top_k", String(Math.max(1, Number(topK) || 1)));

    for (const resumeFile of resumeFiles) {
      formData.append("resumes", resumeFile);
    }

    try {
      const response = await fetch(`${API_BASE_URL}/screen-resumes`, {
        method: "POST",
        body: formData,
      });
      const payload = await response.json();

      if (!response.ok) {
        setErrorText(getErrorMessage(response.status, payload));
        setLoading(false);
        return;
      }

      setResponseData(payload);
      setSelectedIndex(0);
    } catch {
      setErrorText(
        "Request failed. Confirm the backend is running and reachable.",
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="top-bar">
        <div className="brand-block">
          <p className="eyebrow"></p>
          <h1>HR Screening Workspace</h1>
          <p className="subtle">
            Upload resumes, parse against a job description, and review ranked
            candidates.
          </p>
        </div>
      </header>

      <main className="workspace">
        <section className="intake-panel">
          <h2>Screening Input</h2>
          <p></p>

          <form className="screening-form" onSubmit={handleSubmit}>
            <label>
              Job Description
              <textarea
                rows={10}
                value={jobDescription}
                onChange={(event) => setJobDescription(event.target.value)}
                placeholder="Paste full JD text including must-have skills and responsibilities."
              />
            </label>

            <label>
              Resumes
              <input
                type="file"
                multiple
                accept=".pdf,.doc,.docx,.png,.jpg,.jpeg,.tiff,.bmp"
                onChange={(event) =>
                  setResumeFiles(Array.from(event.target.files || []))
                }
              />
            </label>

            <label>
              Candidates to Shortlist
              <input
                type="number"
                min="1"
                value={topK}
                onChange={(event) => setTopK(event.target.value)}
              />
            </label>

            {resumeFiles.length ? (
              <div className="file-list" aria-live="polite">
                {resumeFiles.map((file) => (
                  <p key={`${file.name}-${file.lastModified}`}>{file.name}</p>
                ))}
              </div>
            ) : null}

            {errorText ? <p className="error-text">{errorText}</p> : null}

            <button type="submit" disabled={loading}>
              {loading ? "Screening..." : "Run Screening"}
            </button>
          </form>
        </section>

        <section className="results-panel">
          <div className="results-panel-head">
            <h2>Ranked Results</h2>
            {responseData ? (
              <p>
                Processed {responseData.processed_count} | Failed{" "}
                {responseData.failed_count}
              </p>
            ) : (
              <p></p>
            )}
          </div>

          {responseData &&
          responseData.results &&
          responseData.results.length ? (
            <div className="results-grid">
              <aside className="shortlist">
                {responseData.results.map((item, index) => (
                  <button
                    key={item.source_file}
                    type="button"
                    className={
                      index === selectedIndex
                        ? "shortlist-item active"
                        : "shortlist-item"
                    }
                    onClick={() => setSelectedIndex(index)}
                  >
                    <div className="shortlist-rank">#{index + 1}</div>
                    <div className="shortlist-meta">
                      <strong>
                        {item.resume_data.name || item.source_file}
                      </strong>
                      <p>{item.score.recommendation}</p>
                    </div>
                    <div className="shortlist-score">
                      {item.score.total_score}
                    </div>
                  </button>
                ))}
              </aside>

              {selectedResult ? (
                <article
                  key={selectedResult.source_file}
                  className="candidate-detail"
                >
                  <header className="candidate-head">
                    <div>
                      <p className="eyebrow">Candidate</p>
                      <h3>
                        {selectedResult.resume_data.name ||
                          selectedResult.source_file}
                      </h3>
                      <p>{selectedResult.source_file}</p>
                    </div>
                    <div className="score-total">
                      <span>Total Score</span>
                      <strong>{selectedResult.score.total_score}</strong>
                    </div>
                  </header>

                  <section className="score-breakdown">
                    {SCORE_WEIGHTS.map((scoreItem) => (
                      <ScoreRow
                        key={scoreItem.key}
                        label={scoreItem.label}
                        value={selectedResult.score[scoreItem.key]}
                        max={scoreItem.max}
                      />
                    ))}
                  </section>

                  <section className="detail-row">
                    <h4>Recommendation</h4>
                    <ResultPill text={selectedResult.score.recommendation} />
                  </section>

                  <section className="detail-row">
                    <h4>Recruiter Feedback</h4>
                    <p>
                      {selectedResult.recruiter_feedback ||
                        "No recruiter summary available."}
                    </p>
                  </section>

                  <section className="detail-row">
                    <h4>Matched Skills</h4>
                    <div className="pill-wrap">
                      {selectedResult.score.matched_skills.length ? (
                        selectedResult.score.matched_skills.map((skill) => (
                          <ResultPill key={skill} text={skill} />
                        ))
                      ) : (
                        <p>No matched required skills found.</p>
                      )}
                    </div>
                  </section>

                  <section className="detail-row">
                    <h4>Missing Skills</h4>
                    <div className="pill-wrap">
                      {selectedResult.score.missing_skills.length ? (
                        selectedResult.score.missing_skills.map((skill) => (
                          <ResultPill key={skill} text={skill} />
                        ))
                      ) : (
                        <p>No required skills missing.</p>
                      )}
                    </div>
                  </section>

                  <section className="detail-row">
                    <h4>Profile Signals</h4>
                    <div className="signal-grid">
                      <p>Email: {selectedResult.resume_data.email || "N/A"}</p>
                      <p>Phone: {selectedResult.resume_data.phone || "N/A"}</p>
                      <p>
                        Location: {selectedResult.resume_data.location || "N/A"}
                      </p>
                      <p>
                        Experience:{" "}
                        {selectedResult.resume_data.total_years_experience || 0}{" "}
                        years
                      </p>
                    </div>
                  </section>

                  {selectedResult.score.strengths.length ? (
                    <section className="detail-row">
                      <h4>Strengths</h4>
                      <ul>
                        {selectedResult.score.strengths.map((strength) => (
                          <li key={strength}>{strength}</li>
                        ))}
                      </ul>
                    </section>
                  ) : null}

                  {selectedResult.score.risks.length ? (
                    <section className="detail-row">
                      <h4>Risks</h4>
                      <ul>
                        {selectedResult.score.risks.map((risk) => (
                          <li key={risk}>{risk}</li>
                        ))}
                      </ul>
                    </section>
                  ) : null}
                </article>
              ) : null}
            </div>
          ) : responseData ? (
            <div className="empty-result">
              <h3>No ranked candidates</h3>
              <p>
                The request finished but no valid resumes were scored. Check
                warnings below.
              </p>
            </div>
          ) : (
            <div className="empty-result">
              <h3>Awaiting first screening run</h3>
              <p>Paste a job description, attach resumes, and run screening.</p>
            </div>
          )}

          {responseData &&
          responseData.warnings &&
          responseData.warnings.length ? (
            <section className="warnings">
              <h3>Warnings</h3>
              <ul>
                {responseData.warnings.map((warning, index) => (
                  <li key={`${warning}-${index}`}>{warning}</li>
                ))}
              </ul>
            </section>
          ) : null}
        </section>
      </main>
    </div>
  );
}
