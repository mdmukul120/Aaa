-- শব্দ AI — প্রশ্ন-উত্তর ভাণ্ডার + স্পাইডারের সারি + "সবচেয়ে মিলে যাওয়া" খোঁজার ফাংশন
-- Supabase → SQL Editor-এ একবার চালালেই হয় (বারবার চালালেও ক্ষতি নেই; পুরোনো টেবিল/ডেটা বদলায় না)।

create extension if not exists pg_trgm;

-- ১) প্রশ্ন-উত্তর জোড়া -------------------------------------------------------
create table if not exists public.qa_pairs (
  id          bigint generated always as identity primary key,
  question    text not null,
  answer      text not null,
  q_norm      text not null,                 -- তুলনার জন্য প্রশ্নের এক রূপ (চিহ্ন ছাড়া, ছোট হাতের)
  lang        text not null default 'bn',
  source_url  text,
  site_domain text,
  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now(),
  constraint qa_pairs_q_norm_key unique (q_norm)
);
create index if not exists qa_pairs_q_trgm_idx on public.qa_pairs using gin (q_norm gin_trgm_ops);
create index if not exists qa_pairs_source_idx on public.qa_pairs (source_url);
alter table public.qa_pairs enable row level security;   -- শুধু সার্ভারের service key পড়তে/লিখতে পারে

-- ২) স্পাইডারের সারি (কোন লিংক বাকি, কোনটা শেষ — থেমে গেলে এখান থেকেই আবার শুরু) ------
create table if not exists public.crawl_queue (
  id         bigint generated always as identity primary key,
  root       text not null,                  -- সাইটের হোস্ট (www ছাড়া)
  url        text not null,
  depth      smallint not null default 0,    -- শুরুর লিংক থেকে কত ধাপ দূরে
  status     text not null default 'pending'
             check (status in ('pending', 'done', 'empty', 'failed', 'blocked')),
  note       text,
  found_at   timestamptz not null default now(),
  fetched_at timestamptz,
  constraint crawl_queue_url_key unique (url)
);
create index if not exists crawl_queue_next_idx on public.crawl_queue (root, status, depth, id);
alter table public.crawl_queue enable row level security;

-- ৩) প্রশ্নের সাথে কাছাকাছি জমানো প্রশ্ন (অক্ষর-ত্রয়ীর মিল — বানান ভুল/শব্দের ক্রম বদল সহ্য করে)
create or replace function public.match_qa(q text, lim integer default 12)
returns table(id bigint, question text, answer text, source_url text, site_domain text, lang text, sim real)
language sql stable
set search_path = public
set pg_trgm.similarity_threshold = 0.15
set pg_trgm.word_similarity_threshold = 0.4
as $$
  select p.id, p.question, p.answer, p.source_url, p.site_domain, p.lang,
         greatest(similarity(p.q_norm, q), word_similarity(q, p.q_norm) * 0.9)::real as sim
  from public.qa_pairs p
  where p.q_norm % q or q <% p.q_norm
  order by sim desc, p.id
  limit least(greatest(lim, 1), 50);
$$;

-- ৪) প্রশ্নের শব্দগুলোর যত বেশি সম্ভব যে ডকুমেন্টে আছে (AND নয় — "যত বেশি মেলে তত আগে")
--    terms: ছোট হাতের শব্দ/শব্দমূল; প্রতিটিকে উপসর্গ হিসেবে মেলানো হয় ("ঢাকা" → "ঢাকার"-ও)।
create or replace function public.match_documents(terms text[], lim integer default 8)
returns table(id bigint, title text, url text, site_domain text, lang text, content text, hits integer, rank real)
language sql stable
set search_path = public
as $$
  with t as (
    select term,
           -- য়/ড়/ঢ়-এর দুই ইউনিকোড রূপই খোঁজা
           to_tsquery('simple',
             quote_literal(term) || ':* | ' ||
             quote_literal(replace(replace(replace(term, U&'\09AF\09BC', U&'\09DF'),
                                           U&'\09A1\09BC', U&'\09DC'), U&'\09A2\09BC', U&'\09DD')) || ':*') as tsq
    from (select distinct lower(x) as term from unnest(terms) as x
          where length(x) between 2 and 60 and x !~ '[''\\:&|!()<>*[:space:]]' limit 12) s
  ),
  cand as (
    select d.id, count(*)::integer as hits,
           sum(case when position(t.term in lower(coalesce(d.title, ''))) > 0 then 1 else 0 end) as in_title
    from t
    join public.documents d
      on to_tsvector('simple', coalesce(d.title, '') || ' ' || d.content) @@ t.tsq
    group by d.id
    order by hits desc, in_title desc, d.id desc
    limit 60
  ),
  allq as (select to_tsquery('simple', string_agg(tsq::text, ' | ')) as tsq from t)
  select d.id, d.title, d.url, d.site_domain, d.lang, d.content, c.hits,
         ts_rank_cd(to_tsvector('simple', coalesce(d.title, '') || ' ' || d.content), a.tsq, 1)::real as rank
  from cand c
  join public.documents d on d.id = c.id
  cross join allq a
  order by c.hits desc, c.in_title desc, rank desc
  limit least(greatest(lim, 1), 20);
$$;

-- ৫) স্পাইডারের অবস্থা (একটা সাইটে কয়টা বাকি/শেষ)
create or replace function public.crawl_stats(p_root text)
returns table(pending bigint, done bigint, failed bigint, total bigint)
language sql stable
set search_path = public
as $$
  select count(*) filter (where status = 'pending'),
         count(*) filter (where status in ('done', 'empty')),
         count(*) filter (where status in ('failed', 'blocked')),
         count(*)
  from public.crawl_queue where root = p_root;
$$;

revoke execute on function public.match_qa(text, integer) from public, anon, authenticated;
revoke execute on function public.match_documents(text[], integer) from public, anon, authenticated;
revoke execute on function public.crawl_stats(text) from public, anon, authenticated;
grant execute on function public.match_qa(text, integer) to service_role;
grant execute on function public.match_documents(text[], integer) to service_role;
grant execute on function public.crawl_stats(text) to service_role;
