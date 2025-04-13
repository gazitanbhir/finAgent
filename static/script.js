// Ensure you have a script.js file in the 'static' directory
document.addEventListener('DOMContentLoaded', () => {
    const chatbox = document.getElementById('chatbox');
    const chatForm = document.getElementById('chat-form');
    const commandInput = document.getElementById('command-input');
    const fileInput = document.getElementById('file-input'); // Get file input
    const clearButton = document.getElementById('clear-button'); // Assuming you add this button

    // Function to add messages to the chatbox
    function addMessage(sender, text) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', sender); // 'user' or 'agent'
        // Basic sanitization to prevent HTML injection
        messageDiv.textContent = text;
        chatbox.appendChild(messageDiv);
        chatbox.scrollTop = chatbox.scrollHeight; // Scroll to bottom
    }

    // Handle form submission
    chatForm.addEventListener('submit', async (event) => {
        event.preventDefault(); // Prevent default form submission

        const command = commandInput.value.trim();
        const file = fileInput.files[0]; // Get the selected file

        if (!command && !file) {
            // Maybe show an error message if nothing is entered/uploaded
            console.warn("No command entered or file selected.");
            return;
        }

        // Add user message to chatbox
        let userMessage = command;
        if (file) {
            userMessage += ` (Uploading file: ${file.name})`;
        }
        if (userMessage) { // Only add if there's text or a file indication
             addMessage('user', userMessage);
        }

        // Clear inputs
        commandInput.value = '';
        fileInput.value = ''; // Clear the file input

        // Prepare data using FormData for file upload
        const formData = new FormData();
        formData.append('command', command || ''); // Send empty string if no command but file exists
        if (file) {
            formData.append('file', file, file.name);
        }

        // Add thinking indicator (optional)
        const thinkingDiv = document.createElement('div');
        thinkingDiv.classList.add('message', 'agent', 'thinking');
        thinkingDiv.textContent = 'Agent is thinking...';
        chatbox.appendChild(thinkingDiv);
        chatbox.scrollTop = chatbox.scrollHeight;

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                body: formData // Send FormData directly
                // No 'Content-Type' header needed; browser sets it for FormData
            });

            // Remove thinking indicator
            chatbox.removeChild(thinkingDiv);

            if (!response.ok) {
                // Try to get error detail from response body
                let errorDetail = `HTTP error! Status: ${response.status}`;
                try {
                    const errorData = await response.json();
                    errorDetail = errorData.detail || JSON.stringify(errorData);
                } catch (e) {
                    // If parsing JSON fails, use the status text
                    errorDetail = response.statusText || errorDetail;
                }
                console.error('Chat request failed:', errorDetail);
                addMessage('agent', `Error: ${errorDetail}`);
                return;
            }

            const data = await response.json();
            addMessage('agent', data.response);

        } catch (error) {
             // Remove thinking indicator even on fetch error
             if (chatbox.contains(thinkingDiv)) {
                chatbox.removeChild(thinkingDiv);
             }
            console.error('Error sending chat command:', error);
            addMessage('agent', 'Error: Could not connect to the agent backend.');
        }
    });

    // Handle Clear History Button (if added)
    if (clearButton) {
        clearButton.addEventListener('click', async () => {
            try {
                const response = await fetch('/clear_history', { method: 'POST' });
                if (response.ok || response.status === 204) {
                    // Clear the chatbox visually except for the initial agent message
                    const initialMessage = chatbox.querySelector('.message.agent:first-child');
                    chatbox.innerHTML = ''; // Clear all messages
                    if(initialMessage) {
                         chatbox.appendChild(initialMessage.cloneNode(true)); // Add back the initial greeting
                    }
                    addMessage('agent', 'Chat history cleared.');
                } else {
                     const errorData = await response.json();
                     addMessage('agent', `Error clearing history: ${errorData.detail || response.statusText}`);
                }
            } catch (error) {
                console.error('Error clearing history:', error);
                addMessage('agent', 'Error: Could not reach server to clear history.');
            }
        });
    }
});

// script.js (updated parts only)
// Add these to your existing event listeners

// Updated message creation with rich content support
function addMessage(sender, text) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', sender);
    
    // Create message content structure
    const contentDiv = document.createElement('div');
    contentDiv.classList.add('message-content');
    
    if(sender === 'agent') {
        const header = document.createElement('div');
        header.classList.add('message-header');
        header.textContent = 'AI Assistant';
        contentDiv.appendChild(header);
    }
    
    const textDiv = document.createElement('div');
    textDiv.innerHTML = text.replace(/\n/g, '<br>'); // Preserve line breaks
    contentDiv.appendChild(textDiv);
    
    messageDiv.appendChild(contentDiv);
    chatbox.appendChild(messageDiv);
    
    // Scroll to bottom with smooth behavior
    chatbox.scrollTo({
        top: chatbox.scrollHeight,
        behavior: 'smooth'
    });
}

// Add animation to thinking indicator
function showThinkingIndicator() {
    const thinkingDiv = document.createElement('div');
    thinkingDiv.classList.add('message', 'agent', 'thinking');
    thinkingDiv.textContent = 'Analyzing your request';
    chatbox.appendChild(thinkingDiv);
    return thinkingDiv;
}

// In your submit handler, replace thinking indicator creation with:
const thinkingDiv = showThinkingIndicator();