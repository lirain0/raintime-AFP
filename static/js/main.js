// AFP预测平台 - 主要JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // 导航栏移动端菜单（如果需要）
    initMobileNav();
    
    // 自动隐藏Flash消息
    initFlashMessages();
    
    // 序列输入验证
    initSequenceValidation();
    
    // 初始化阈值控制
    initThresholdControl();
});

// 阈值控制初始化
function initThresholdControl() {
    const slider = document.getElementById('threshold-slider');
    const valueDisplay = document.getElementById('threshold-value');
    const resetBtn = document.getElementById('threshold-reset');
    
    if (!slider || !valueDisplay) return;
    
    // 获取当前阈值
    fetch('/api/threshold')
        .then(res => res.json())
        .then(data => {
            slider.value = data.threshold;
            valueDisplay.textContent = data.threshold.toFixed(2);
        })
        .catch(err => console.error('获取阈值失败:', err));
    
    // 滑块变化时更新阈值（使用箭头函数避免this问题）
    let debounceTimer;
    slider.addEventListener('input', function(e) {
        const newThreshold = parseFloat(e.target.value);
        valueDisplay.textContent = newThreshold.toFixed(2);
        
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
            fetch('/api/threshold', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ threshold: newThreshold })
            })
            .then(res => res.json())
            .then(data => {
                if (data.error) {
                    console.error('更新阈值失败:', data.error);
                } else {
                    console.log('阈值已更新:', data.threshold);
                }
            })
            .catch(err => console.error('更新阈值失败:', err));
        }, 300);
    });
    
    // 重置按钮
    if (resetBtn) {
        resetBtn.addEventListener('click', function() {
            fetch('/api/threshold/reset', {
                method: 'POST'
            })
            .then(res => res.json())
            .then(data => {
                slider.value = data.threshold;
                valueDisplay.textContent = data.threshold.toFixed(2);
                console.log('阈值已重置:', data.threshold);
            })
            .catch(err => console.error('重置阈值失败:', err));
        });
    }
}

// 移动端导航
function initMobileNav() {
    // 如果需要移动端菜单，可以在这里添加
    // 目前使用响应式CSS隐藏菜单
}

// Flash消息自动隐藏
function initFlashMessages() {
    const flashes = document.querySelectorAll('.flash');
    flashes.forEach(flash => {
        setTimeout(() => {
            flash.style.opacity = '0';
            flash.style.transform = 'translateX(100%)';
            setTimeout(() => flash.remove(), 300);
        }, 5000);
    });
}

// 序列输入验证
function initSequenceValidation() {
    const validAminoAcids = 'ACDEFGHIKLMNPQRSTVWY';
    
    const sequenceInputs = document.querySelectorAll('.sequence-input, #sequence');
    sequenceInputs.forEach(input => {
        input.addEventListener('input', function() {
            // 转换为大写
            let value = this.value.toUpperCase();
            
            // 过滤无效字符
            value = value.split('').filter(char => validAminoAcids.includes(char)).join('');
            
            // 更新输入
            if (value !== this.value) {
                this.value = value;
            }
        });
    });
}

// 工具函数：防抖
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// 工具函数：格式化数字
function formatNumber(num, decimals = 2) {
    return Number(num).toFixed(decimals);
}

// 工具函数：格式化百分比
function formatPercent(num, decimals = 2) {
    return (num * 100).toFixed(decimals) + '%';
}

// 工具函数：复制到剪贴板
async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        return true;
    } catch (err) {
        console.error('复制失败:', err);
        return false;
    }
}

// 工具函数：下载文件
function downloadFile(content, filename, type = 'text/plain') {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// 工具函数：解析FASTA格式
function parseFASTA(content) {
    const sequences = [];
    const lines = content.split('\n');
    let currentSeq = { id: '', sequence: '' };
    
    for (const line of lines) {
        if (line.startsWith('>')) {
            if (currentSeq.sequence) {
                sequences.push({ ...currentSeq });
            }
            currentSeq = {
                id: line.substring(1).trim(),
                sequence: ''
            };
        } else {
            currentSeq.sequence += line.trim();
        }
    }
    
    if (currentSeq.sequence) {
        sequences.push(currentSeq);
    }
    
    return sequences;
}

// 工具函数：生成唯一ID
function generateId() {
    return Date.now().toString(36) + Math.random().toString(36).substr(2);
}

// 工具函数：计算氨基酸组成
function calculateComposition(sequence) {
    const aaCounts = {};
    const validAA = 'ACDEFGHIKLMNPQRSTVWY';
    
    for (const aa of validAA) {
        aaCounts[aa] = 0;
    }
    
    for (const aa of sequence.toUpperCase()) {
        if (validAA.includes(aa)) {
            aaCounts[aa]++;
        }
    }
    
    const total = sequence.length;
    const composition = {};
    
    for (const aa of validAA) {
        composition[aa] = (aaCounts[aa] / total * 100).toFixed(2);
    }
    
    return composition;
}

// 工具函数：计算分子量（近似）
function calculateMW(sequence) {
    const aaWeights = {
        'A': 89.09, 'C': 121.16, 'D': 133.10, 'E': 147.13,
        'F': 165.19, 'G': 75.07, 'H': 155.16, 'I': 131.17,
        'K': 146.19, 'L': 131.17, 'M': 149.21, 'N': 132.12,
        'P': 115.13, 'Q': 146.15, 'R': 174.20, 'S': 105.09,
        'T': 119.12, 'V': 117.15, 'W': 204.23, 'Y': 181.19
    };
    
    let mw = 18.015; // 水分子
    for (const aa of sequence.toUpperCase()) {
        if (aaWeights[aa]) {
            mw += aaWeights[aa];
        }
    }
    
    return mw.toFixed(2);
}

// 页面滚动动画
function initScrollAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');
            }
        });
    }, { threshold: 0.1 });
    
    document.querySelectorAll('.feature-card, .stat-item').forEach(el => {
        observer.observe(el);
    });
}

// 平滑滚动
function smoothScrollTo(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.scrollIntoView({ behavior: 'smooth' });
    }
}

// 初始化DNA动画
function initDNAAnimation() {
    const container = document.querySelector('.dna-animation');
    if (!container) return;
    
    // DNA动画已通过CSS实现
    // 这里可以添加额外的交互效果
}

// 导出函数供其他脚本使用
window.AFPUtils = {
    debounce,
    formatNumber,
    formatPercent,
    copyToClipboard,
    downloadFile,
    parseFASTA,
    generateId,
    calculateComposition,
    calculateMW,
    smoothScrollTo
};