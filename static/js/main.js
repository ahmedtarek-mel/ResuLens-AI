document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const form = document.getElementById('analysis-form');
    const demoToggle = document.getElementById('demo-toggle');
    const dropArea = document.getElementById('drop-area');
    const fileInput = dropArea.querySelector('.file-input');
    const fileNameDisplay = document.getElementById('file-name-display');
    const wordCount = document.getElementById('word-count');
    const jobDescInput = document.getElementById('job-description');
    const spinner = document.getElementById('loading-spinner');
    const resultsContainer = document.getElementById('results-container');

    // Configuration Sliders
    const sliderSemantic = document.getElementById('slider-semantic');
    const sliderSkill = document.getElementById('slider-skill');
    const sliderKeyword = document.getElementById('slider-keyword');
    const valSemantic = document.getElementById('val-semantic');
    const valSkill = document.getElementById('val-skill');
    const valKeyword = document.getElementById('val-keyword');

    // Mobile Menu
    const mobileMenuBtn = document.getElementById('mobile-menu-btn');
    const sidebar = document.getElementById('sidebar');
    const sidebarClose = document.getElementById('sidebar-close');

    // Store weights globally
    window.analysisWeights = {
        semantic: 0.40,
        skill: 0.35,
        keyword: 0.25
    };

    // Store last analysis scores for recalculation
    window.lastAnalysisScores = null;

    // Recalculate overall score based on current weights
    function recalculateScore() {
        if (!window.lastAnalysisScores) return;

        const { semantic, skill, keyword } = window.analysisWeights;
        const { semanticScore, skillScore, keywordScore } = window.lastAnalysisScores;

        // Normalize weights to sum to 1.0
        const total = semantic + skill + keyword;
        const normSemantic = semantic / total;
        const normSkill = skill / total;
        const normKeyword = keyword / total;

        // Calculate new overall score
        const newScore = (semanticScore * normSemantic) + (skillScore * normSkill) + (keywordScore * normKeyword);
        const roundedScore = Math.round(newScore * 10) / 10;

        // Update gauge chart
        Plotly.update('gauge-chart', { value: [roundedScore] });

        // Update grade
        let grade, gradeText, color, bg;
        if (roundedScore >= 90) { grade = 'A+'; gradeText = 'Excellent'; color = '#10B981'; bg = '#ECFDF5'; }
        else if (roundedScore >= 80) { grade = 'A'; gradeText = 'Great Match'; color = '#10B981'; bg = '#ECFDF5'; }
        else if (roundedScore >= 70) { grade = 'B+'; gradeText = 'Good Match'; color = '#3B82F6'; bg = '#EFF6FF'; }
        else if (roundedScore >= 60) { grade = 'B'; gradeText = 'Fair Match'; color = '#3B82F6'; bg = '#EFF6FF'; }
        else if (roundedScore >= 50) { grade = 'C+'; gradeText = 'Needs Improvement'; color = '#F59E0B'; bg = '#FFFBEB'; }
        else if (roundedScore >= 40) { grade = 'C'; gradeText = 'Below Average'; color = '#F59E0B'; bg = '#FFFBEB'; }
        else { grade = 'D'; gradeText = 'Needs Work'; color = '#EF4444'; bg = '#FEF2F2'; }

        const gradeCircle = document.getElementById('grade-circle');
        const gradeTextEl = document.getElementById('grade-text');
        if (gradeCircle && gradeTextEl) {
            gradeCircle.textContent = grade;
            gradeTextEl.textContent = gradeText;
            gradeCircle.style.color = color;
            gradeCircle.style.backgroundColor = bg;
            gradeCircle.style.border = `2px solid ${color}20`;
            gradeTextEl.style.color = color;
        }
    }

    // Configuration Slider Handlers
    function updateSliderValue(slider, display, key) {
        const value = parseFloat(slider.value);
        display.textContent = value.toFixed(2);
        window.analysisWeights[key] = value;
        recalculateScore(); // Live recalculation
    }

    if (sliderSemantic && sliderSkill && sliderKeyword) {
        sliderSemantic.addEventListener('input', () => {
            updateSliderValue(sliderSemantic, valSemantic, 'semantic');
        });

        sliderSkill.addEventListener('input', () => {
            updateSliderValue(sliderSkill, valSkill, 'skill');
        });

        sliderKeyword.addEventListener('input', () => {
            updateSliderValue(sliderKeyword, valKeyword, 'keyword');
        });
    }


    // Mobile Menu Handlers
    if (mobileMenuBtn && sidebar && sidebarClose) {
        mobileMenuBtn.addEventListener('click', () => {
            sidebar.classList.add('active');
            document.body.style.overflow = 'hidden';
        });

        sidebarClose.addEventListener('click', () => {
            sidebar.classList.remove('active');
            document.body.style.overflow = '';
        });

        // Close on outside click
        document.addEventListener('click', (e) => {
            if (sidebar.classList.contains('active') &&
                !sidebar.contains(e.target) &&
                !mobileMenuBtn.contains(e.target)) {
                sidebar.classList.remove('active');
                document.body.style.overflow = '';
            }
        });
    }

    // 1. Drag & Drop File Upload

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    const inputCard = dropArea.closest('.input-card');
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => inputCard.classList.add('highlight'), false);
    });
    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => inputCard.classList.remove('highlight'), false);
    });

    dropArea.addEventListener('drop', handleDrop);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    fileInput.addEventListener('change', function () {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            fileNameDisplay.textContent = `✓ ${file.name}`;
            fileNameDisplay.classList.remove('hidden');
        }
    }

    // 2. Word Count
    jobDescInput.addEventListener('input', () => {
        const text = jobDescInput.value;
        const count = text.trim().split(/\s+/).filter(w => w.length > 0).length;
        wordCount.textContent = `${count} words`;
    });

    // 3. Demo Toggle
    demoToggle.addEventListener('change', (e) => {
        const isDemo = e.target.checked;
        if (isDemo) {
            dropArea.style.opacity = '0.5';
            dropArea.style.pointerEvents = 'none';
            jobDescInput.disabled = true;
            jobDescInput.placeholder = "Demo data loaded internally...";
            fileNameDisplay.textContent = "✓ Sample_Resume.pdf (Demo)";
            fileNameDisplay.classList.remove('hidden');
        } else {
            dropArea.style.opacity = '1';
            dropArea.style.pointerEvents = 'auto';
            jobDescInput.disabled = false;
            jobDescInput.placeholder = "Paste the complete job description here...";
            fileNameDisplay.classList.add('hidden');
            fileInput.value = ''; // clear
        }
    });

    // 4. Form Submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const isDemo = demoToggle.checked;
        const formData = new FormData();

        if (isDemo) {
            formData.append('source', 'demo');
        } else {
            const files = fileInput.files;
            const jobDesc = jobDescInput.value;

            if (files.length === 0) {
                alert('Please upload a resume.');
                return;
            }
            if (jobDesc.length < 50) {
                alert('Job description is too short.');
                return;
            }

            formData.append('source', 'upload');
            formData.append('resume_file', files[0]);
            formData.append('job_description', jobDesc);
        }

        // Show Spinner
        spinner.classList.remove('hidden');
        resultsContainer.classList.add('hidden'); // Hide previous results

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.error || 'Analysis failed');
            }

            const data = await response.json();
            renderDashboard(data);

        } catch (error) {
            alert('Error: ' + error.message);
        } finally {
            spinner.classList.add('hidden');
        }
    });

    // 5. Render Dashboard
    function renderDashboard(data) {
        resultsContainer.classList.remove('hidden');

        // Store scores for live recalculation when sliders change
        window.lastAnalysisScores = {
            semanticScore: data.semantic_score,
            skillScore: data.skill_score,
            keywordScore: data.keyword_score
        };

        // Render Gauge

        const gaugeData = [{
            type: "indicator",
            mode: "gauge+number",
            value: data.overall_score,
            number: { suffix: "%", font: { size: 48, color: "#1E293B", family: "Inter" } },
            gauge: {
                axis: { range: [0, 100], tickcolor: "#CBD5E1" },
                bar: { color: "#4F46E5" },
                bgcolor: "#F1F5F9",
                borderwidth: 0,
                steps: [
                    { range: [0, 40], color: "#FEF2F2" },
                    { range: [40, 60], color: "#FFFBEB" },
                    { range: [60, 80], color: "#EFF6FF" },
                    { range: [80, 100], color: "#F0FDF4" }
                ]
            }
        }];
        const gaugeLayout = { height: 250, margin: { t: 25, b: 25, l: 25, r: 25 } };
        Plotly.newPlot('gauge-chart', gaugeData, gaugeLayout, { displayModeBar: false });

        // Grade Badge
        const gradeCircle = document.getElementById('grade-circle');
        const gradeText = document.getElementById('grade-text');
        gradeCircle.textContent = data.grade;
        gradeText.textContent = data.grade_text;

        // Set colors based on grade (simple logic)
        let color = '#64748B';
        let bg = '#F8FAFC';

        if (['A', 'A+'].includes(data.grade)) { color = '#10B981'; bg = '#ECFDF5'; }
        else if (['B', 'B+'].includes(data.grade)) { color = '#3B82F6'; bg = '#EFF6FF'; }
        else if (['C', 'C+'].includes(data.grade)) { color = '#F59E0B'; bg = '#FFFBEB'; }
        else { color = '#EF4444'; bg = '#FEF2F2'; }

        gradeCircle.style.color = color;
        gradeCircle.style.backgroundColor = bg;
        gradeCircle.style.borderColor = color + '20'; // light border
        gradeCircle.style.border = `2px solid ${color}20`;
        gradeText.style.color = color;

        // Render Radar
        const radarData = [{
            type: 'scatterpolar',
            r: [data.semantic_score, data.skill_score, data.keyword_score, data.semantic_score],
            theta: ['Semantic', 'Skills', 'Keywords', 'Semantic'],
            fill: 'toself',
            fillcolor: 'rgba(79, 70, 229, 0.1)',
            line: { color: '#4F46E5' }
        }];

        const radarLayout = {
            polar: {
                radialaxis: { visible: true, range: [0, 100] }
            },
            margin: { t: 20, b: 20, l: 40, r: 40 },
            height: 300
        };
        Plotly.newPlot('radar-chart', radarData, radarLayout, { displayModeBar: false });

        // Render Bar Breakdown
        const cats = ['Semantic', 'Skills', 'Keywords'];
        const scores = [data.semantic_score, data.skill_score, data.keyword_score];
        const barColors = scores.map(s => s >= 70 ? '#4F46E5' : (s < 40 ? '#F59E0B' : '#6366F1'));

        const barData = [{
            type: 'bar',
            x: scores,
            y: cats,
            orientation: 'h',
            marker: { color: barColors },
            text: scores.map(s => s.toFixed(1) + '%'),
            textposition: 'auto',
            textfont: { color: '#1E293B', size: 12 }
        }];

        const barLayout = {
            xaxis: {
                range: [0, 100],
                tickmode: 'linear',
                tick0: 0,
                dtick: 20,
                tickfont: { size: 12, color: '#64748B' },
                showgrid: true,
                gridcolor: '#E2E8F0'
            },
            yaxis: {
                tickfont: { size: 12, color: '#1E293B' }
            },
            margin: { t: 10, b: 40, l: 80, r: 30 },
            height: 160
        };

        Plotly.newPlot('bar-chart', barData, barLayout, { displayModeBar: false });

        // Skills Lists
        renderList('list-matched', data.matched_skills, '✅');
        renderList('list-missing', data.missing_skills, '❌');
        renderList('list-extra', data.extra_skills, 'ℹ️');

        document.getElementById('count-matched').textContent = data.matched_skills.length;
        document.getElementById('count-missing').textContent = data.missing_skills.length;
        document.getElementById('count-extra').textContent = data.extra_skills.length;

        // Keywords Cloud
        const cloudContainer = document.getElementById('keywords-cloud');
        cloudContainer.innerHTML = '';
        if (data.top_keywords) {
            data.top_keywords.slice(0, 12).forEach(([word, _]) => {
                const tag = document.createElement('div');
                tag.className = 'keyword-tag';
                tag.textContent = word;
                cloudContainer.appendChild(tag);
            });
        }

        // Recommendations with expand/collapse
        const recContainer = document.getElementById('recommendations-list');
        recContainer.innerHTML = '';

        const maxVisibleRecs = 3;
        let recsExpanded = false;

        function renderRecommendations() {
            recContainer.innerHTML = '';
            const visibleRecs = recsExpanded ? data.recommendations : data.recommendations.slice(0, maxVisibleRecs);

            visibleRecs.forEach(rec => {
                const div = document.createElement('div');

                let typeClass = 'rec-note';
                let iconStr = 'ℹ️';
                let titleStr = 'Note';

                if (rec.includes('Strong') || rec.includes('Excellent')) {
                    typeClass = 'rec-success'; iconStr = '✅'; titleStr = 'Keep it up';
                } else if (rec.includes('Add') || rec.includes('Include')) {
                    typeClass = 'rec-action'; iconStr = '➕'; titleStr = 'Action Item';
                } else if (rec.includes('Consider') || rec.includes('Improve')) {
                    typeClass = 'rec-suggest'; iconStr = '💡'; titleStr = 'Suggestion';
                }

                const cleanRec = rec.replace(/[✅⚠️💡]/g, '').trim();

                div.className = `rec-card ${typeClass}`;
                div.innerHTML = `
                    <span class="icon">${iconStr}</span>
                    <div>
                        <div class="title">${titleStr}</div>
                        <div class="text">${cleanRec}</div>
                    </div>
                `;
                recContainer.appendChild(div);
            });

            // Add expand/collapse toggle if more than max
            if (data.recommendations.length > maxVisibleRecs) {
                const toggle = document.createElement('div');
                toggle.className = 'rec-toggle';
                toggle.style.cssText = 'color:#4F46E5; font-size:0.9rem; cursor:pointer; font-weight:500; padding:0.75rem; text-align:center; margin-top:0.5rem;';

                if (recsExpanded) {
                    toggle.textContent = '▲ Show less';
                } else {
                    toggle.textContent = `+ ${data.recommendations.length - maxVisibleRecs} more recommendations...`;
                }

                toggle.addEventListener('click', () => {
                    recsExpanded = !recsExpanded;
                    renderRecommendations();
                });

                recContainer.appendChild(toggle);
            }
        }

        renderRecommendations();


        // Scroll to results
        resultsContainer.scrollIntoView({ behavior: 'smooth' });
    }

    function renderList(id, items, icon) {
        const el = document.getElementById(id);
        el.innerHTML = '';
        if (!items || items.length === 0) {
            el.innerHTML = '<span style="color:#94a3b8; font-style:italic;">None</span>';
            return;
        }

        const maxVisible = 8;
        let expanded = false;

        function render() {
            el.innerHTML = '';
            const visibleItems = expanded ? items : items.slice(0, maxVisible);

            visibleItems.forEach(item => {
                const span = document.createElement('span');
                span.innerHTML = `${icon} ${item}`;
                el.appendChild(span);
            });

            if (items.length > maxVisible) {
                const toggle = document.createElement('span');
                toggle.className = 'see-more-btn';
                toggle.style.cssText = 'color:#4F46E5; font-size:0.85rem; cursor:pointer; font-weight:500; display:block; margin-top:0.5rem;';

                if (expanded) {
                    toggle.textContent = '▲ Show less';
                } else {
                    toggle.textContent = `+ ${items.length - maxVisible} more...`;
                }

                toggle.addEventListener('click', () => {
                    expanded = !expanded;
                    render();
                });

                el.appendChild(toggle);
            }
        }

        render();
    }


});
