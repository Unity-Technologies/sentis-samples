using System;
using System.Collections.Generic;
using Unity.AppUI.Redux;
using Unity.AppUI.UI;
using Unity.InferenceEngine.Samples.TTS.Inference;
using Unity.InferenceEngine.Samples.TTS.State;
using UnityEngine;
using UnityEngine.UIElements;

namespace Unity.InferenceEngine.Samples.TTS.UI
{
    [UxmlElement]
    public partial class App : VisualElement
    {
        TextArea m_InputArea;
        Dropdown m_VoiceDropdown;
        ActionButton m_GenerateSpeech;
        AudioPlayer m_AudioPlayer;
        TouchSliderFloat m_SpeedSlider;

        AppStoreManager m_StoreManager;
        IDisposableSubscription m_StoreSubscription;

        List<KokoroHandler.Voice> m_Voices;

        public App()
        {
            RegisterCallback<AttachToPanelEvent>(OnAttachToPanel);
            RegisterCallback<DetachFromPanelEvent>(OnDetachFromPanel);
            m_StoreManager = new AppStoreManager();
        }

        void OnAttachToPanel(AttachToPanelEvent evt)
        {
            m_InputArea = this.Q<TextArea>(classes: "tts-input-area");
            m_InputArea.RegisterValueChangingCallback(OnInputValueChanging);

            m_Voices = KokoroHandler.GetVoices();
            m_VoiceDropdown = this.Q<Dropdown>(classes: "tts-voice-dropdown");
            m_VoiceDropdown.RegisterValueChangedCallback(OnVoiceChanged);
            m_VoiceDropdown.sourceItems = m_Voices;
            m_VoiceDropdown.bindItem += DropDownBindItem;
            m_VoiceDropdown.makeItem += DropDownMakeItem;
            m_VoiceDropdown.selectedIndex = 0;
            m_VoiceDropdown.Refresh();

            m_GenerateSpeech = this.Q<ActionButton>(classes: "tts-generate-button");
            m_GenerateSpeech.clickable.clicked += OnGenerateSpeechClicked;

            m_AudioPlayer = this.Q<AudioPlayer>(classes: "tts-audio-player");
            m_AudioPlayer.Initialize(m_StoreManager);

            m_SpeedSlider = this.Q<TouchSliderFloat>(classes: "tts-speed-slider");
            m_SpeedSlider.RegisterValueChangedCallback(OnSpeedChanged);

            m_StoreSubscription = m_StoreManager.Store.Subscribe( AppSlice.Name, (AppState state) => state.Status, OnStatusChanged);
            UpdateVisuals(m_StoreManager.Store.GetState<AppState>(AppSlice.Name).Status);
        }


        void DropDownBindItem(DropdownItem arg1, int arg2)
        {
            if (arg2 >= 0 && arg2 < m_Voices.Count)
            {
                arg1.label = m_Voices[arg2].Name;
            }
        }

        DropdownItem DropDownMakeItem()
        {
            var item = new DropdownItem
            {
                style =
                {
                    height = 20,
                    unityTextAlign = TextAnchor.MiddleLeft,
                    paddingLeft = 4
                }
            };
            return item;
        }

        void OnDetachFromPanel(DetachFromPanelEvent evt)
        {
            foreach (var voice in m_Voices)
            {
                voice?.Dispose();
            }
            m_Voices.Clear();
            m_Voices = null;

            m_InputArea.UnregisterValueChangingCallback(OnInputValueChanging);
            m_InputArea = null;
            m_VoiceDropdown.UnregisterValueChangedCallback(OnVoiceChanged);
            m_VoiceDropdown = null;
            m_GenerateSpeech.clickable.clicked -= OnGenerateSpeechClicked;
            m_GenerateSpeech = null;
            m_AudioPlayer = null;
            m_StoreSubscription?.Dispose();
        }

        void OnInputValueChanging(ChangingEvent<string> evt)
        {
            m_StoreManager.Store.Dispatch(m_StoreManager.SetInputText.Invoke(evt.newValue));
        }

        void OnVoiceChanged(ChangeEvent<IEnumerable<int>> evt)
        {
            if (evt.newValue is List<int> { Count: > 0 } selectedIndices)
            {
                var selectedIndex = selectedIndices[0];
                if (selectedIndex >= 0 && selectedIndex < m_Voices.Count)
                {
                    m_StoreManager.Store.Dispatch(m_StoreManager.SetVoice.Invoke(m_Voices[selectedIndex]));
                }
            }
        }

        async void OnGenerateSpeechClicked()
        {
            var action = m_StoreManager.GenerateSpeech.Invoke();
            await m_StoreManager.Store.DispatchAsyncThunk(action);
        }

        void OnStatusChanged(AppState.GenerationStatus state)
        {
            UpdateVisuals(state);
        }

        void OnSpeedChanged(ChangeEvent<float> evt)
        {
            m_StoreManager.Store.Dispatch(m_StoreManager.SetSpeed.Invoke(evt.newValue));
        }

        void UpdateVisuals(AppState.GenerationStatus status)
        {
            m_AudioPlayer.style.display = status == AppState.GenerationStatus.Completed ? DisplayStyle.Flex : DisplayStyle.None;
            m_GenerateSpeech.SetEnabled(status != AppState.GenerationStatus.Pending);
            m_SpeedSlider.SetEnabled(status != AppState.GenerationStatus.Pending);
            m_InputArea.SetEnabled(status != AppState.GenerationStatus.Pending);
            m_VoiceDropdown.SetEnabled(status != AppState.GenerationStatus.Pending);
        }
    }
}
