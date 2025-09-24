using System.Threading.Tasks;
using Unity.InferenceEngine.Samples.TTS.State;
using UnityEngine.UIElements;
using Unity.AppUI.Redux;
using Unity.AppUI.UI;
using Unity.InferenceEngine.Samples.TTS.Utils;
using UnityEngine;

namespace Unity.InferenceEngine.Samples.TTS.UI
{
    [UxmlElement]
    public partial class AudioPlayer : VisualElement
    {
        AppStoreManager m_StoreManager;
        AudioSource m_AudioSource;

        IconButton m_PlayButton;
        IconButton m_SaveButton;
        Text m_TimeStamp;
        SliderFloat m_AudioProgress;

        public AudioPlayer()
        {
            RegisterCallback<AttachToPanelEvent>(OnAttachToPanel);
            RegisterCallback<DetachFromPanelEvent>(OnDetachFromPanel);
        }

        void OnAttachToPanel(AttachToPanelEvent evt)
        {
            m_AudioSource = GameObject.Find("App").GetComponent<AudioSource>();
            m_PlayButton = this.Q<IconButton>(classes: "play-button");
            m_PlayButton.clickable.clicked += OnPlayButtonClicked;
            m_SaveButton = this.Q<IconButton>(classes: "download-button");
            m_SaveButton.clickable.clicked += OnSaveButtonClicked;
            m_TimeStamp = this.Q<Text>(classes: "time-text");
            m_AudioProgress = this.Q<SliderFloat>(classes: "progress-bar");
            m_AudioProgress.RegisterValueChangingCallback(OnSliderValueChanging);

            m_AudioProgress.RegisterCallback<PointerDownEvent>(_ => m_AudioSource.Pause(), TrickleDown.TrickleDown);
            m_AudioProgress.RegisterCallback<PointerUpEvent>(_ =>  Play(), TrickleDown.TrickleDown);
        }

        void OnSliderValueChanging(ChangingEvent<float> evt)
        {
            m_AudioSource.time = evt.newValue / 100f * m_AudioSource.clip.length;
            UpdateVisuals();
        }

        void OnDetachFromPanel(DetachFromPanelEvent evt)
        {
            m_PlayButton.clickable.clicked -= OnPlayButtonClicked;
            m_SaveButton.clickable.clicked -= OnSaveButtonClicked;
        }

        void OnPlayButtonClicked()
        {
            if(m_AudioSource.isPlaying)
                m_AudioSource.Pause();
            else
                Play();
        }

        async Task Play()
        {
            m_AudioSource.Play();
            while (m_AudioSource.isPlaying)
            {
                UpdateVisuals();
                await Task.Yield();
            }

            UpdateVisuals();
        }

        public void Initialize(AppStoreManager storeManager)
        {
            m_StoreManager = storeManager;
            m_StoreManager.Store.Subscribe( AppSlice.Name, (AppState state) => state.AudioClip, OnAudioClipChanged);
        }

        void OnAudioClipChanged(AudioClip clip)
        {
            m_AudioSource.clip = clip;
            UpdateVisuals();
        }

        void UpdateVisuals()
        {
            m_AudioProgress.SetValueWithoutNotify(m_AudioSource.time / m_AudioSource.clip.length * 100f);
            m_TimeStamp.text = $"{m_AudioSource.time:0:00} / {m_AudioSource.clip.length:0:00}";
            m_PlayButton.icon = m_AudioSource.isPlaying ? "pause" : "play";

            #if !UNITY_EDITOR
            m_SaveButton.SetEnabled(false);
            #endif
        }

        void OnSaveButtonClicked()
        {
            #if UNITY_EDITOR
            //Open save file dialog
            var path = UnityEditor.EditorUtility.SaveFilePanel("Save Audio", "", "audio.wav", "wav");
            var state = m_StoreManager.Store.GetState<AppState>(AppSlice.Name);
            WavUtils.WriteFloatWav(path, state.Waveform.DownloadToArray());
            #endif
        }
    }
}
