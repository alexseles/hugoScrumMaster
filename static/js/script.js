document.addEventListener("DOMContentLoaded", () => {
  const chatLog = document.getElementById("chat-log");
  const userInput = document.getElementById("user-input");
  const sendBtn = document.getElementById("send-btn");
  const fileInput = document.getElementById("file-input");
  const fileUploadBtn = document.getElementById("file-upload-btn");
  const fileNameDisplay = document.getElementById("file-name");

  let chatHistory = [];

  // Função para sanitizar texto e evitar XSS
  const sanitizeHTML = (str) => {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  };

  // Adiciona evento de clique ao botão de clip para abrir o seletor de arquivos
  fileUploadBtn.addEventListener("click", () => {
    fileInput.click();
  });

  // Mostra o nome do arquivo quando um é selecionado
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
      fileNameDisplay.textContent = sanitizeHTML(fileInput.files[0].name);
    } else {
      fileNameDisplay.textContent = "";
    }
  });

  // Função para adicionar uma mensagem à interface do chat
  const addMessage = (sender, message) => {
    const typingIndicatorContainer = document.getElementById("typing-indicator-container");
    if (typingIndicatorContainer) {
      typingIndicatorContainer.remove();
    }

    const messageElement = document.createElement("div");
    messageElement.classList.add("message", sender);
    messageElement.setAttribute("role", "log"); // Acessibilidade

    const contentElement = document.createElement("div");
    contentElement.classList.add("message-content");
    
    const pElement = document.createElement("p");
    // Usa textContent com sanitização para evitar XSS
    pElement.innerHTML = message
      .replace(/\n/g, "<br>")
      .replace(/\*\*(.*?)\*\*/g, "<b>$1</b>"); // Suporte básico a markdown
    contentElement.appendChild(pElement);
    
    messageElement.appendChild(contentElement);
    chatLog.appendChild(messageElement);
    chatLog.scrollTop = chatLog.scrollHeight;
  };

  const showTypingIndicator = () => {
    const messageElement = document.createElement("div");
    messageElement.classList.add("message", "ai");
    messageElement.id = "typing-indicator-container";
    messageElement.setAttribute("aria-live", "polite"); // Acessibilidade
    messageElement.innerHTML = `
      <div class="typing-indicator" id="typing-indicator">
        <span></span><span></span><span></span>
      </div>
    `;
    chatLog.appendChild(messageElement);
    chatLog.scrollTop = chatLog.scrollHeight;
  };

  const sendMessage = async () => {
    const messageText = userInput.value.trim();
    const file = fileInput.files[0];

    if (messageText === "" && !file) return;

    // Constrói a mensagem a ser exibida
    let userMessageDisplay = sanitizeHTML(messageText);
    if (file) {
      userMessageDisplay = `<b>Ficheiro: ${sanitizeHTML(file.name)}</b><br>${userMessageDisplay}`;
    }
    addMessage("user", userMessageDisplay);
    chatHistory.push({ role: "user", content: messageText });

    // Limpa a interface
    userInput.value = "";
    fileInput.value = "";
    fileNameDisplay.textContent = "";
    userInput.disabled = true;
    sendBtn.disabled = true;

    showTypingIndicator();

    const formData = new FormData();
    formData.append("message", messageText);
    formData.append("history", JSON.stringify(chatHistory));
    if (file) {
      formData.append("file", file);
    }

    try {
      const response = await fetch("/chat", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `Erro de HTTP! Estado: ${response.status}`);
      }

      const data = await response.json();
      const aiResponse = data.response;

      // Verifica se o backend retornou um erro relacionado a quotas
      if (aiResponse.includes("429") || aiResponse.includes("quota")) {
        addMessage("ai", "O servidor atingiu o limite de quota da API. Por favor, tente novamente mais tarde ou contate o administrador.");
      } else {
        addMessage("ai", sanitizeHTML(aiResponse));
        chatHistory.push({ role: "ai", content: aiResponse });
      }

    } catch (error) {
      console.error("Erro ao comunicar com o servidor:", error);
      addMessage("ai", `Erro: ${sanitizeHTML(error.message)}. Verifique a consola do servidor para mais detalhes.`);
    } finally {
      userInput.disabled = false;
      sendBtn.disabled = false;
      userInput.focus();
    }
  };

  sendBtn.addEventListener("click", sendMessage);
  userInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault(); // Evita quebra de linha com Shift+Enter
      sendMessage();
    }
  });

  const addInitialMessage = () => {
    const welcomeMessage = "Olá! Eu sou o Hugo, o seu Scrum Master Copilot. Como posso ajudá-lo a aplicar os princípios ágeis hoje?";
    addMessage("ai", welcomeMessage);
    chatHistory.push({ role: "ai", content: welcomeMessage });
    userInput.disabled = false;
    userInput.placeholder = "Pergunte sobre eventos, papéis, artefactos...";
    userInput.focus();
  };

  addInitialMessage();
});