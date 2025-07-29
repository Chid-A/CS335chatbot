let currentLat = 48.8566;
let currentLon = 2.3522;

let selectedVoice = null;

// Load voices and pick a friendly female English voice
function loadVoices() {
  const voices = window.speechSynthesis.getVoices();
  console.log("Available voices:", voices.map(v => v.name));

  // Filter female English voices
  const femaleVoices = voices.filter(voice =>
    voice.lang.startsWith('en') &&
    (voice.name.toLowerCase().includes('female') ||
      voice.name.toLowerCase().includes('woman') ||
      voice.name.toLowerCase().includes('zira') ||      // Windows female voice
      voice.name.toLowerCase().includes('samantha') ||  // Mac female voice
      voice.name.toLowerCase().includes('google uk english female') || // Chrome female voice
      voice.name.toLowerCase().includes('google us english female'))
  );

  if (femaleVoices.length > 0) {
    selectedVoice = femaleVoices[0];
  } else {
    // fallback to any English voice
    selectedVoice = voices.find(voice => voice.lang.startsWith('en')) || voices[0];
  }
}

// Some browsers load voices asynchronously
window.speechSynthesis.onvoiceschanged = loadVoices;
setTimeout(loadVoices, 500);

function speakText(text) {
  if (window.speechSynthesis.speaking) {
    window.speechSynthesis.cancel();
  }
  const utterance = new SpeechSynthesisUtterance(text);

  if (selectedVoice) {
    utterance.voice = selectedVoice;
  }

  // Slightly higher pitch and normal rate for a friendly tone
  utterance.pitch = 1.1;
  utterance.rate = 1;

  window.speechSynthesis.speak(utterance);
}

async function getBotResponse(userMessage) {
  try {
    const response = await fetch("http://127.0.0.1:5000/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: userMessage,
        latitude: currentLat,
        longitude: currentLon,
      }),
    });
    const data = await response.json();
    currentLat = data.latitude;
    currentLon = data.longitude;

    return data.reply;
  } catch (error) {
    console.error("Backend error:", error);
    return "⚠️ Could not reach the server.";
  }
}

document.getElementById("chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();

  const input = document.getElementById("user-input");
  const userMessage = input.value.trim();
  if (!userMessage) return;

  addMessageToChat(userMessage, "user-message");
  input.value = "";

  // Add typing indicator
  const typingDiv = document.createElement("div");
  typingDiv.className = "bot-message typing-indicator";
  typingDiv.textContent = "👩 Typing...";
  const chatBox = document.getElementById("chat-box");
  chatBox.appendChild(typingDiv);
  chatBox.scrollTop = chatBox.scrollHeight;

  // Simulate delay (1.2 - 2 seconds)
  const delay = Math.random() * (2000 - 1200) + 1200;
  await new Promise((resolve) => setTimeout(resolve, delay));

  const botReply = await getBotResponse(userMessage);

  // Remove typing indicator
  chatBox.removeChild(typingDiv);

  // Add actual bot reply
  addMessageToChat("👩 " + botReply, "bot-message");

  // Speak response
  speakText(botReply);
});

// Initial greeting on page load
window.addEventListener("DOMContentLoaded", async () => {
  const initialGreeting = "Hello! I'm your European Travel Assistant. Ask me about famous landmarks, delicious cuisine, or unique cultures all across Europe. How may I help you today?";

  // Show typing indicator
  const chatBox = document.getElementById("chat-box");
  const typingDiv = document.createElement("div");
  typingDiv.className = "bot-message typing-indicator";
  typingDiv.textContent = "👩 Typing...";
  chatBox.appendChild(typingDiv);
  chatBox.scrollTop = chatBox.scrollHeight;


  // Simulate delay (0.8 - 1 second)
  const delay = Math.random() * (1000 - 800) + 1200;
  await new Promise((resolve) => setTimeout(resolve, delay));
  chatBox.removeChild(typingDiv);
  addMessageToChat("👩 " + initialGreeting, "bot-message");
  speakText(initialGreeting);
});

function addMessageToChat(message, className) {
  const chatBox = document.getElementById("chat-box");
  const messageDiv = document.createElement("div");
  messageDiv.className = className;
  messageDiv.textContent = message;
  chatBox.appendChild(messageDiv);
  chatBox.scrollTop = chatBox.scrollHeight;
}


const chatWindow = document.getElementById("chat-window");

// Add handles for all 4 sides
const handles = {
  right: document.createElement("div"),
  left: document.createElement("div"),
  top: document.createElement("div"),
  bottom: document.createElement("div"),
  topLeft: document.createElement("div"),
  topRight: document.createElement("div"),
  bottomLeft: document.createElement("div"),
  bottomRight: document.createElement("div"),
};
handles.right.className = "resize-handle handle-right";
handles.left.className = "resize-handle handle-left";
handles.top.className = "resize-handle handle-top";
handles.bottom.className = "resize-handle handle-bottom";
handles.topLeft.className = "resize-handle handle-top-left";
handles.topRight.className = "resize-handle handle-top-right";
handles.bottomLeft.className = "resize-handle handle-bottom-left";
handles.bottomRight.className = "resize-handle handle-bottom-right";
chatWindow.appendChild(handles.right);
chatWindow.appendChild(handles.left);
chatWindow.appendChild(handles.top);
chatWindow.appendChild(handles.bottom);
chatWindow.appendChild(handles.topLeft);
chatWindow.appendChild(handles.topRight);
chatWindow.appendChild(handles.bottomLeft);
chatWindow.appendChild(handles.bottomRight);

let isDragging = false;
let isResizing = false;
let resizeDir = null;
let dragOffsetX = 0;
let dragOffsetY = 0;
let initialRect = null;

// Dragging
chatWindow.addEventListener("mousedown", (e) => {
  if (e.target.classList.contains("resize-handle")) return;
  isDragging = true;
  const rect = chatWindow.getBoundingClientRect();
  dragOffsetX = e.clientX - rect.left;
  dragOffsetY = e.clientY - rect.top;
  initialRect = rect;
});

document.addEventListener("mousemove", (e) => {
  if (isDragging) {
    let newLeft = e.clientX - dragOffsetX;
    let newTop = e.clientY - dragOffsetY;
    newLeft = Math.max(0, Math.min(window.innerWidth - chatWindow.offsetWidth, newLeft));
    newTop = Math.max(0, Math.min(window.innerHeight - chatWindow.offsetHeight, newTop));
    chatWindow.style.position = "fixed";
    chatWindow.style.left = `${newLeft}px`;
    chatWindow.style.top = `${newTop}px`;
  } else if (isResizing && resizeDir) {
    const minWidth = 250, minHeight = 200;
    const rect = initialRect;
    const right = rect.right;
    const bottom = rect.bottom;
    const left = rect.left;
    const top = rect.top;

    if (resizeDir === "right") {
      let maxWidth = window.innerWidth - left;
      let newWidth = Math.max(minWidth, Math.min(maxWidth, e.clientX - left));
      chatWindow.style.width = `${newWidth}px`;
    } else if (resizeDir === "left") {
      let minLeft = 0;
      let maxLeft = right - minWidth;
      let newLeft = Math.max(minLeft, Math.min(maxLeft, e.clientX));
      let newWidth = right - newLeft;
      
      // Prevent overflow to the right
      if (newLeft + newWidth > window.innerWidth) {
        newWidth = window.innerWidth - newLeft;
      }
      chatWindow.style.left = `${newLeft}px`;
      chatWindow.style.width = `${newWidth}px`;
      return;
    } else if (resizeDir === "top") {
      let minTop = 0;
      let maxTop = bottom - minHeight;
      let newTop = Math.max(minTop, Math.min(maxTop, e.clientY));
      let newHeight = bottom - newTop;

      // Prevent overflow to the bottom
      if (newTop + newHeight > window.innerHeight) {
        newHeight = window.innerHeight - newTop;
      }
      chatWindow.style.top = `${newTop}px`;
      chatWindow.style.height = `${newHeight}px`;
      return;
    } else if (resizeDir === "bottom") {
      let maxHeight = window.innerHeight - top;
      let newHeight = Math.max(minHeight, Math.min(maxHeight, e.clientY - top));
      chatWindow.style.height = `${newHeight}px`;
      return;
    }
  }
});

document.addEventListener("mouseup", () => {
  isDragging = false;
  isResizing = false;
  resizeDir = null;
});

// Resizing
Object.entries(handles).forEach(([dir, handle]) => {
  handle.addEventListener("mousedown", (e) => {
    isResizing = true;
    resizeDir = dir;
    const rect = chatWindow.getBoundingClientRect();
    initialRect = rect;
    e.stopPropagation();
  });
});


//sidebar script
const sidebar = document.querySelector('.left-sidebar');
const sidebarToggle = document.getElementById('sidebar-toggle');

sidebarToggle.addEventListener('click', () => {
  sidebar.classList.toggle('collapsed');
  sidebarToggle.classList.toggle('collapsed');


  // Change arrow direction
  sidebarToggle.innerHTML = sidebar.classList.contains('collapsed') ? '&#x25B6;' : '&#x25C0;';
});
