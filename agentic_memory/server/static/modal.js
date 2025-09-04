/**
 * Synthwave Modal System
 * A reusable modal component for the JAM system
 */

class SynthModal {
    constructor() {
        this.activeModal = null;
        this.init();
    }
    
    init() {
        // Wait for DOM to be ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.createModalRoot());
        } else {
            this.createModalRoot();
        }
    }
    
    createModalRoot() {
        // Create modal container if it doesn't exist
        if (!document.getElementById('modal-root')) {
            const modalRoot = document.createElement('div');
            modalRoot.id = 'modal-root';
            document.body.appendChild(modalRoot);
        }
    }
    
    /**
     * Show a confirmation modal
     * @param {Object} options - Modal configuration
     * @param {string} options.title - Modal title
     * @param {string} options.message - Main message
     * @param {string} options.submessage - Additional message (optional)
     * @param {string} options.type - Modal type: 'info', 'warning', 'danger', 'success'
     * @param {string} options.confirmText - Confirm button text
     * @param {string} options.cancelText - Cancel button text
     * @param {Function} options.onConfirm - Callback for confirm
     * @param {Function} options.onCancel - Callback for cancel
     * @param {boolean} options.showWarningBox - Show warning box (optional)
     * @param {string} options.warningText - Warning box text (optional)
     */
    confirm(options) {
        const defaults = {
            title: 'Confirm Action',
            message: 'Are you sure?',
            submessage: '',
            type: 'info',
            confirmText: 'Confirm',
            cancelText: 'Cancel',
            onConfirm: () => {},
            onCancel: () => {},
            showWarningBox: false,
            warningText: ''
        };
        
        const config = { ...defaults, ...options };
        
        // Create modal HTML
        const modalHtml = `
            <div class="modal-overlay active" id="synth-modal">
                <div class="modal-container ${config.type === 'danger' ? 'shake' : ''}">
                    <div class="modal-header">
                        <h3 class="modal-title">${this.escapeHtml(config.title)}</h3>
                        <button class="modal-close" onclick="synthModal.close()">×</button>
                    </div>
                    <div class="modal-body">
                        ${this.getIcon(config.type)}
                        <div class="modal-message">${this.escapeHtml(config.message)}</div>
                        ${config.submessage ? `<div class="modal-submessage">${this.escapeHtml(config.submessage)}</div>` : ''}
                        ${config.showWarningBox ? `
                            <div class="${config.type === 'danger' ? 'modal-danger-box' : 'modal-warning-box'}">
                                ${this.escapeHtml(config.warningText)}
                            </div>
                        ` : ''}
                    </div>
                    <div class="modal-footer">
                        <button class="modal-btn modal-btn-cancel" onclick="synthModal.handleCancel()">
                            ${this.escapeHtml(config.cancelText)}
                        </button>
                        <button class="modal-btn ${config.type === 'danger' ? 'modal-btn-danger' : 'modal-btn-confirm'}" 
                                onclick="synthModal.handleConfirm()">
                            ${this.escapeHtml(config.confirmText)}
                        </button>
                    </div>
                </div>
            </div>
        `;
        
        // Store callbacks
        this._onConfirm = config.onConfirm;
        this._onCancel = config.onCancel;
        
        // Ensure modal root exists
        this.createModalRoot();
        
        // Add to DOM
        document.getElementById('modal-root').innerHTML = modalHtml;
        this.activeModal = document.getElementById('synth-modal');
        
        // Add escape key handler
        this._escapeHandler = (e) => {
            if (e.key === 'Escape') {
                this.close();
            }
        };
        document.addEventListener('keydown', this._escapeHandler);
        
        // Add click outside handler
        this.activeModal.addEventListener('click', (e) => {
            if (e.target === this.activeModal) {
                this.close();
            }
        });
    }
    
    /**
     * Show a multi-step confirmation modal for dangerous actions
     * @param {Object} options - Modal configuration
     */
    async confirmDangerous(options) {
        return new Promise((resolve) => {
            // First confirmation
            this.confirm({
                title: options.title || ' Warning',
                message: options.message || 'This action cannot be undone!',
                submessage: options.submessage,
                type: 'warning',
                confirmText: 'I Understand',
                cancelText: 'Cancel',
                showWarningBox: true,
                warningText: options.warningText || 'This action is permanent and cannot be reversed.',
                onConfirm: () => {
                    // Second confirmation
                    setTimeout(() => {
                        this.confirm({
                            title: '🚨 Final Confirmation',
                            message: options.finalMessage || 'Please confirm once more:',
                            submessage: 'This is your last chance to cancel.',
                            type: 'danger',
                            confirmText: options.finalConfirmText || 'Yes, Delete Everything',
                            cancelText: 'Cancel',
                            showWarningBox: true,
                            warningText: 'YOU ARE ABOUT TO PERMANENTLY DELETE ALL DATA!',
                            onConfirm: () => {
                                resolve(true);
                                if (options.onConfirm) options.onConfirm();
                            },
                            onCancel: () => {
                                resolve(false);
                                if (options.onCancel) options.onCancel();
                            }
                        });
                    }, 100);
                },
                onCancel: () => {
                    resolve(false);
                    if (options.onCancel) options.onCancel();
                }
            });
        });
    }
    
    /**
     * Show an alert modal
     * @param {Object} options - Modal configuration
     */
    alert(options) {
        const defaults = {
            title: 'Alert',
            message: '',
            type: 'info',
            buttonText: 'OK',
            onClose: () => {}
        };
        
        const config = { ...defaults, ...options };
        
        const modalHtml = `
            <div class="modal-overlay active" id="synth-modal">
                <div class="modal-container">
                    <div class="modal-header">
                        <h3 class="modal-title">${this.escapeHtml(config.title)}</h3>
                        <button class="modal-close" onclick="synthModal.close()">×</button>
                    </div>
                    <div class="modal-body">
                        ${this.getIcon(config.type)}
                        <div class="modal-message">${this.escapeHtml(config.message)}</div>
                    </div>
                    <div class="modal-footer">
                        <button class="modal-btn modal-btn-confirm" onclick="synthModal.close()">
                            ${this.escapeHtml(config.buttonText)}
                        </button>
                    </div>
                </div>
            </div>
        `;
        
        // Store callback
        this._onClose = config.onClose;
        
        // Ensure modal root exists
        this.createModalRoot();
        
        document.getElementById('modal-root').innerHTML = modalHtml;
        this.activeModal = document.getElementById('synth-modal');
        
        // Add escape key handler
        this._escapeHandler = (e) => {
            if (e.key === 'Escape') {
                this.close();
            }
        };
        document.addEventListener('keydown', this._escapeHandler);
    }
    
    /**
     * Show a custom content modal
     * @param {Object} options - Modal configuration
     * @param {string} options.title - Modal title
     * @param {string} options.content - Custom HTML content
     * @param {string} options.width - Modal width (optional)
     * @param {Function} options.onClose - Callback when modal is closed
     */
    custom(options) {
        const defaults = {
            title: 'Modal',
            content: '',
            width: '600px',
            onClose: () => {}
        };
        
        const config = { ...defaults, ...options };
        
        const modalHtml = `
            <div class="modal-overlay active" id="synth-modal">
                <div class="modal-container" style="width: ${config.width}; max-width: 90%;">
                    <div class="modal-header">
                        <h3 class="modal-title">${this.escapeHtml(config.title)}</h3>
                        <button class="modal-close" onclick="synthModal.close()">×</button>
                    </div>
                    <div class="modal-body" style="padding: 0;">
                        ${config.content}
                    </div>
                </div>
            </div>
        `;
        
        // Store callback
        this._onClose = config.onClose;
        
        // Ensure modal root exists
        this.createModalRoot();
        
        document.getElementById('modal-root').innerHTML = modalHtml;
        this.activeModal = document.getElementById('synth-modal');
        
        // Add escape key handler
        this._escapeHandler = (e) => {
            if (e.key === 'Escape') {
                this.close();
            }
        };
        document.addEventListener('keydown', this._escapeHandler);
        
        // Add click outside handler
        this.activeModal.addEventListener('click', (e) => {
            if (e.target === this.activeModal) {
                this.close();
            }
        });
    }
    
    handleConfirm() {
        if (this._onConfirm) {
            this._onConfirm();
        }
        this.close();
    }
    
    handleCancel() {
        if (this._onCancel) {
            this._onCancel();
        }
        this.close();
    }
    
    close() {
        // Call onClose callback if it exists
        if (this._onClose) {
            this._onClose();
        }
        
        if (this.activeModal) {
            this.activeModal.classList.remove('active');
            setTimeout(() => {
                if (this.activeModal) {
                    this.activeModal.remove();
                    this.activeModal = null;
                }
            }, 300);
        }
        
        // Remove event listener
        if (this._escapeHandler) {
            document.removeEventListener('keydown', this._escapeHandler);
            this._escapeHandler = null;
        }
        
        // Clear callbacks
        this._onConfirm = null;
        this._onCancel = null;
        this._onClose = null;
    }
    
    getIcon(type) {
        const icons = {
            'info': '<div class="modal-icon info">ℹ️</div>',
            'warning': '<div class="modal-icon warning">⚠️</div>',
            'danger': '<div class="modal-icon danger">🚨</div>',
            'success': '<div class="modal-icon success">✅</div>'
        };
        return icons[type] || icons['info'];
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Create global instance and attach to window when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function() {
        window.synthModal = new SynthModal();
    });
} else {
    window.synthModal = new SynthModal();
}